from torch.utils.data import Dataset
import random
from datasets import load_dataset
from tqdm import tqdm
import json
import pickle
import os

class ICL_dataset(Dataset):
    def __init__(self, args):
        self.args = args
        ## sentiment classification (binary)
        if args.data=="cr": ## 전자제품 리뷰 감정 분류 데이터셋 train 3394 test 376
            data = load_dataset("SetFit/CR")
            self.train_data = data['train']
            self.dev_data = data['test']
            self.train_data = {i:(t,l) for i, (t, l) in enumerate(zip(self.train_data['text'], self.train_data['label']))}
            self.dev_data = {i:(t,l) for i, (t, l) in enumerate(zip(self.dev_data['text'], self.dev_data['label']))}
            self.id2label = {0: 'negative', 1: 'positive'}
            self.instruction = "Given electronics review text, predict its sentiment. positive or negative?"
        elif args.data=="sst2": ## 영화 리뷰 감정 분류 데이터셋 train 67349 validation 872 test 1821(X) 
            data = load_dataset('glue', 'sst2')
            self.train_data = data['train']
            self.dev_data = data['validation']
            self.train_data = {i:(t,l) for i, (t, l) in enumerate(zip(self.train_data['sentence'], self.train_data['label']))}
            self.dev_data = {i:(t,l) for i, (t, l) in enumerate(zip(self.dev_data['sentence'], self.dev_data['label']))}
            self.id2label = {0: 'negative', 1: 'positive'}
            self.instruction = "Given movie review text, predict its sentiment. positive or negative?"
        elif args.data=="TweetEval_Hate": ## 트윗 혐오 발언 포함여부 train 9000 valid 2970 test 1000
            data = load_dataset('tweet_eval', 'hate')
            self.train_data = data['train']
            self.dev_data = data['test']
            self.train_data = {i:(t,l) for i, (t, l) in enumerate(zip(self.train_data['text'], self.train_data['label']))}
            self.dev_data = {i:(t,l) for i, (t, l) in enumerate(zip(self.dev_data['text'], self.dev_data['label']))}
            self.id2label = {0: 'no', 1: 'yes'}
            self.instruction = "Given Tweet text, predict whether it is hate text. yes or no?"
        ## sentiment classification (polarity)
        elif args.data=='sst5': ## 영화 리뷰 감정 분류 데이터셋 train 8544 validation 1101 test 2210 
            data = load_dataset("SetFit/sst5")
            self.train_data = data['train']
            self.dev_data = data['test']
            self.train_data = {i:(t,l) for i, (t, l) in enumerate(zip(self.train_data['text'], self.train_data['label']))}
            self.dev_data = {i:(t,l) for i, (t, l) in enumerate(zip(self.dev_data['text'], self.dev_data['label']))}
            self.id2label = {0: 'terrible', 1: 'bad', 2:'okay', 3:'good', 4:'great'}
            self.instruction = "Given review text, predict its sentiment. great, good, okay, bad, terrible?"
        ## NLI Classification  
        elif args.data=='rte': ## NLI 형 데이터셋 train 2490 validation 277 test 3000(X)
            data = load_dataset("glue", 'rte')
            self.train_data = data['train']
            self.dev_data = data['validation']
            self.train_data = {i:([t1,t2],l) for i, (t1, t2, l) in enumerate(zip(self.train_data['sentence1'], self.train_data['sentence2'], self.train_data['label']))}
            self.dev_data =  {i:([t1,t2],l) for i, (t1, t2, l) in enumerate(zip(self.dev_data['sentence1'], self.dev_data['sentence2'], self.dev_data['label']))}
            self.id2label = {0:'yes', 1:'no'}
        elif args.data=='wnli':
            data = load_dataset("glue", 'wnli') ## NLI 형 데이터셋 train 635 validation 71 test 146(x)
            self.train_data = data['train']
            self.dev_data = data['validation']
            self.train_data = {i:([t1,t2],l) for i, (t1, t2, l) in enumerate(zip(self.train_data['sentence1'], self.train_data['sentence2'], self.train_data['label']))}
            self.dev_data =  {i:([t1,t2],l) for i, (t1, t2, l) in enumerate(zip(self.dev_data['sentence1'], self.dev_data['sentence2'], self.dev_data['label']))}
            self.id2label = {0:'no', 1:'yes'}
        ## Question type Classification
        elif args.data=="trec": ## 질문 유형 분류 데이터셋 train 5452 test 500
            data = load_dataset('trec')
            self.train_data = data['train']
            self.dev_data = data['test']
            self.train_data = {i:(t,l) for i, (t, l) in enumerate(zip(self.train_data['text'], self.train_data['coarse_label']))}
            self.dev_data = {i:(t,l) for i, (t, l) in enumerate(zip(self.dev_data['text'], self.dev_data['coarse_label']))}
            self.id2label = {0:'abbreviation', 1:'entity', 2:'description', 3:'human', 4:'location', 5:'number'}
        ## Topic Classification
        elif args.data=="agnews": ## 기사 주제 분류 데이터셋 train 120000 test 7600
            data = load_dataset("ag_news")
            self.train_data = data['train']
            self.dev_data = data['test']
            self.train_data = {i:(t,l) for i, (t, l) in enumerate(zip(self.train_data['text'], self.train_data['label']))}
            self.dev_data = {i:(t,l) for i, (t, l) in enumerate(zip(self.dev_data['text'], self.dev_data['label']))}
            self.id2label = {0: 'world', 1: 'sports', 2:'business', 3:'sci/tech'}
        self.label2id = {v:k for k,v in self.id2label.items()}
        self.id2verb = list(self.label2id.keys())
        print(f"load {args.data} datasets")

        if self.args.data == "sst2" or self.args.data == "cr":
            self.template=self.template_binary_sentiment
        elif self.args.data == "trec":
            self.template=self.template_trec
        elif self.args.data == "rte" or self.args.data == "wnli":
            self.template = self.template_nli
        elif self.args.data =="sst5":
            self.template = self.template_sst5
        elif self.args.data=="agnews":
            self.template = self.template_agnews
        elif self.args.data=="TweetEval_Hate":
            self.template = self.template_hate

        if self.args.max_train_size<len(self.train_data.keys()):
            org_train_len = len(self.train_data.keys())
            self.train_data = self.limit_dataset_size(self.train_data, self.args.max_train_size)
            print(f"Initial train dataset size : {org_train_len} -> {len(self.train_data.keys())}")
        else:
            print(f"Initial train dataset size :{len(self.train_data.keys())}")
        
        if self.args.max_dev_size<len(self.dev_data.keys()):
            org_dev_len = len(self.dev_data.keys())
            self.dev_data = self.limit_dataset_size(self.dev_data, self.args.max_dev_size, False)
            print(f"Initial dev dataset size : {org_dev_len} -> {len(self.dev_data.keys())}")
        else:
            print(f"Initial dev dataset size :{len(self.dev_data.keys())}")

    ## only when limited size is smaller than total dataset size
    def limit_dataset_size(self, data, limited_data_size, train=True):
        if train:
            if os.path.isfile(f'./limited_dataset_idx/{self.args.data}_train{limited_data_size}.json'):
                with open(f'./limited_dataset_idx/{self.args.data}_train{limited_data_size}.json', 'r') as f:
                    limited_idx = json.load(f)
            else:
                os.makedirs(f'./limited_dataset_idx/', exist_ok=True) 
                idx = list(data.keys())
                limited_idx = random.sample(idx, limited_data_size)
                with open(f'./limited_dataset_idx/{self.args.data}_train{limited_data_size}.json', 'w') as f:
                    json.dump(limited_idx,f)
            return {k : data[k] for k in limited_idx}
        else:
            if os.path.isfile(f'./limited_dataset_idx/{self.args.data}_dev{limited_data_size}.json'):
                with open(f'./limited_dataset_idx/{self.args.data}_dev{limited_data_size}.json', 'r') as f:
                    limited_idx = json.load(f)
            else:
                os.makedirs(f'./limited_dataset_idx/', exist_ok=True) 
                idx = list(data.keys())
                limited_idx = random.sample(idx, limited_data_size)
                with open(f'./limited_dataset_idx/{self.args.data}_dev{limited_data_size}.json', 'w') as f:
                    json.dump(limited_idx,f)
            return {k : data[k] for k in limited_idx}
        
    
    def template_binary_sentiment(self, sentence, label=None, n_filler = 0, mode='train'):
        if mode == 'train':
            return f"Review:{sentence}\nSentiment:" + "..."*n_filler+ f"{self.id2label[label]}\n"
        else:
            return f"Review:{sentence}\nSentiment:" + "..."*n_filler 
    def template_hate(self, sentence, label=None, n_filler = 0, mode='train'):
        if mode == 'train':
            return f"Text:{sentence}\nIs this hate text?:" + "..."* n_filler+f"{self.id2label[label]}\n"
        else:
            return f"Text:{sentence}\nIs this hate text?:" + "..."*n_filler 
    def template_trec(self, sentence, label=None, n_filler = 0, mode='train'):
        if mode == 'train':
            return f"Question:{sentence}\nType:" + "..."* n_filler+f"{self.id2label[label]}\n"
        else:
            return f"Question:{sentence}\nType:" + "..."*n_filler 
    def template_nli(self, sentence, label=None, n_filler = 0, mode='train'):
        if mode == 'train':
            return f"Sentence 1:{sentence[0]}\nSentence 2:{sentence[1]}\nEntailment?:" + "..."* n_filler +f"{self.id2label[label]}\n"
        else:
            return f"Sentence 1:{sentence[0]}\nSentence 2:{sentence[1]}\nEntailment?:" + "..."* n_filler
    def template_sst5(self, sentence, label=None, n_filler = 0, mode='train'):
        if mode == 'train':
            return f"Text:{sentence}\nSentiment:" + "..."* n_filler+f"{self.id2label[label]}\n"
        else:
            return f"Text:{sentence}\nSentiment:" + "..."*n_filler 
    def template_agnews(self, sentence, label=None, n_filler = 0, mode='train'):
        if mode == 'train':
            return f"Article:{sentence}\nTopic:" + "..."* n_filler +f"{self.id2label[label]}\n"
        else:
            return f"Article:{sentence}\nTopic:" + "..."*n_filler 

    def __len__(self):
        return len(self.dev_data.keys()) 
    
    def __getitem__(self, idx):
        i, (t, l) = list(self.dev_data.items())[idx]
        prompt = ''
        prompt+=self.template(t, n_filler = self.args.n_filler, mode = 'inference')
        label = l
        return i, prompt, label