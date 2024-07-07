from torch.utils.data import Dataset
import random
from datasets import load_dataset, load_from_disk
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
            self.args.single = True
        elif args.data=="sst2": ## 영화 리뷰 감정 분류 데이터셋 train 67349 validation 872 test 1821(X) 
            data = load_dataset('glue', 'sst2')
            self.train_data = data['train']
            self.dev_data = data['validation']
            self.train_data = {i:(t,l) for i, (t, l) in enumerate(zip(self.train_data['sentence'], self.train_data['label']))}
            self.dev_data = {i:(t,l) for i, (t, l) in enumerate(zip(self.dev_data['sentence'], self.dev_data['label']))}
            self.id2label = {0: 'negative', 1: 'positive'}
            self.instruction = "Given movie review text, predict its sentiment. positive or negative?"
            self.args.single = True
        ## hate speech detection (binary)
        elif args.data=="TweetEval_Hate": ## 트윗 혐오 발언 포함여부 train 9000 valid 2970 test 1000
            data = load_dataset('tweet_eval', 'hate')
            self.train_data = data['train']
            self.dev_data = data['test']
            self.train_data = {i:(t,l) for i, (t, l) in enumerate(zip(self.train_data['text'], self.train_data['label']))}
            self.dev_data = {i:(t,l) for i, (t, l) in enumerate(zip(self.dev_data['text'], self.dev_data['label']))}
            self.id2label = {0: 'no', 1: 'yes'}
            self.instruction = "Given Tweet text, predict whether it is hate text. yes or no?"
            self.args.single = True
        elif args.data=="hate_speech18": ## 혐오 발언 분류 데이터셋 train only 10944 -> train 8755 test 2189 (0.2 split) -> train 8562 test 2141 (relation, idk label 제외)
            data = self.split_dataset(self.args, d_name ="hate_speech18" , test_size = 0.2)
            self.train_data = data['train']
            self.dev_data = data['test']
            self.train_data = {i:(t,l) for i, t, l in zip(self.train_data['idx'], self.train_data['text'], self.train_data['label'])}
            self.dev_data = {i:(t,l) for i, t, l in zip(self.dev_data['idx'], self.dev_data['text'], self.dev_data['label'])}
            self.id2label = {0: 'no', 1: 'yes'}
            self.instruction = "Given speech text, predict whether it is hate text. yes or no?"
            self.args.single = True
        ## sentiment classification (polarity)
        elif args.data=='sst5': ## 영화 리뷰 감정 분류 데이터셋 train 8544 validation 1101 test 2210 
            data = load_dataset("SetFit/sst5")
            self.train_data = data['train']
            self.dev_data = data['test']
            self.train_data = {i:(t,l) for i, (t, l) in enumerate(zip(self.train_data['text'], self.train_data['label']))}
            self.dev_data = {i:(t,l) for i, (t, l) in enumerate(zip(self.dev_data['text'], self.dev_data['label']))}
            self.id2label = {0: 'terrible', 1: 'bad', 2:'okay', 3:'good', 4:'great'}
            self.instruction = "Given review text, predict its sentiment. great, good, okay, bad, terrible?"
            self.args.single = True
        ## NLI Classification  
        elif args.data=='rte': ## NLI 형 데이터셋 train 2490 validation 277 test 3000(X)
            data = load_dataset("glue", 'rte')
            self.train_data = data['train']
            self.dev_data = data['validation']
            self.train_data = {i:([t1,t2],l) for i, (t1, t2, l) in enumerate(zip(self.train_data['sentence1'], self.train_data['sentence2'], self.train_data['label']))}
            self.dev_data =  {i:([t1,t2],l) for i, (t1, t2, l) in enumerate(zip(self.dev_data['sentence1'], self.dev_data['sentence2'], self.dev_data['label']))}
            self.id2label = {0:'yes', 1:'no'}
            self.instruction = ""
            self.args.single = False
        elif args.data=='wnli': ## NLI 형 데이터셋 train 635 validation 71 test 146(x)
            data = load_dataset("glue", 'wnli') 
            self.train_data = data['train']
            self.dev_data = data['validation']
            self.train_data = {i:([t1,t2],l) for i, (t1, t2, l) in enumerate(zip(self.train_data['sentence1'], self.train_data['sentence2'], self.train_data['label']))}
            self.dev_data =  {i:([t1,t2],l) for i, (t1, t2, l) in enumerate(zip(self.dev_data['sentence1'], self.dev_data['sentence2'], self.dev_data['label']))}
            self.id2label = {0:'no', 1:'yes'}
            self.instruction = ""
            self.args.single = False
        elif args.data=="mnli":
            data = load_dataset("glue", 'mnli') 
            self.train_data = data['train']
            self.dev_data = data['validation_matched']
            self.train_data = {i:([t1,t2],l) for i, (t1, t2, l) in enumerate(zip(self.train_data['premise'], self.train_data['hypothesis'], self.train_data['label']))}
            self.dev_data =  {i:([t1,t2],l) for i, (t1, t2, l) in enumerate(zip(self.dev_data['premise'], self.dev_data['hypothesis'], self.dev_data['label']))}
            self.id2label = {0:'yes', 1:'neither', 2:'no'}
            self.instruction = 'In this task, you are given two sentences. Indicate if the first sentence clearly entails the second sentence (i.e., one can conclude the 2nd sentence by reading the 1st one). Indicate your answer with \'yes\' if the first sentence entails the second sentence. If the first sentence contradicts the second sentence, answer with \'no\'. Otherwise, answer with \'neither\'.'
            self.args.single = False         
        elif args.data=="snli":
            data = load_dataset('stanfordnlp/snli') 
            self.train_data = data['train']
            self.train_data = self.train_data.filter(lambda example : example['label']!=-1)
            self.dev_data = data['test']
            self.dev_data = self.dev_data.filter(lambda example : example['label']!=-1)
            self.train_data = {i:([t1,t2],l) for i, (t1, t2, l) in enumerate(zip(self.train_data['premise'], self.train_data['hypothesis'], self.train_data['label']))}
            self.dev_data =  {i:([t1,t2],l) for i, (t1, t2, l) in enumerate(zip(self.dev_data['premise'], self.dev_data['hypothesis'], self.dev_data['label']))}
            self.id2label = {0:'yes', 1:'neither', 2:'no'}
            self.instruction = 'In this task, you are given two sentences. Indicate if the first sentence clearly entails the second sentence (i.e., one can conclude the 2nd sentence by reading the 1st one). Indicate your answer with \'yes\' if the first sentence entails the second sentence. If the first sentence contradicts the second sentence, answer with \'no\'. Otherwise, answer with \'neither\'.'
        elif args.data=="mrpc": ## 두 문장의 논리적 일치 여부 train 3668 val 408 test 1725
            data = load_dataset("glue","mrpc")
            self.train_data = data['train']
            self.dev_data = data['test']
            self.train_data = {i:([t1,t2],l) for i, (t1, t2, l) in enumerate(zip(self.train_data['sentence1'], self.train_data['sentence2'], self.train_data['label']))}
            self.dev_data =  {i:([t1,t2],l) for i, (t1, t2, l) in enumerate(zip(self.dev_data['sentence1'], self.dev_data['sentence2'], self.dev_data['label']))}
            self.id2label = {0:'no', 1:'yes'}
            self.instruction = ""
            self.args.single = False        
        elif args.data=="mqp": ## 의학적 질문쌍 유사도 여부 train only 3048
            data = self.split_dataset(self.args, d_name='medical_questions_pairs', test_size=0.2)
            self.train_data = data['train']
            self.dev_data = data['test']
            self.train_data = {i:([t1,t2],l) for i, t1, t2, l in zip(self.train_data['idx'], self.train_data['question_1'], self.train_data['question_2'], self.train_data['label'])}
            self.dev_data =  {i:([t1,t2],l) for i, t1, t2, l in zip(self.dev_data['idx'], self.dev_data['question_1'], self.dev_data['question_2'], self.dev_data['label'])}
            self.id2label = {0:'no', 1:'yes'}
            self.instruction = ""
            self.args.single = False 
        ## Question type Classification
        elif args.data=="trec": ## 질문 유형 분류 데이터셋 train 5452 test 500
            data = load_dataset('trec')
            self.train_data = data['train']
            self.dev_data = data['test']
            self.train_data = {i:(t,l) for i, (t, l) in enumerate(zip(self.train_data['text'], self.train_data['coarse_label']))}
            self.dev_data = {i:(t,l) for i, (t, l) in enumerate(zip(self.dev_data['text'], self.dev_data['coarse_label']))}
            self.id2label = {0:'abbreviation', 1:'entity', 2:'description', 3:'human', 4:'location', 5:'number'}
            self.instruction = ""
            self.args.single = True
        ## Topic Classification
        elif args.data=="agnews": ## 기사 주제 분류 데이터셋 train 120000 test 7600
            data = load_dataset("ag_news")
            self.train_data = data['train']
            self.dev_data = data['test']
            self.train_data = {i:(t,l) for i, (t, l) in enumerate(zip(self.train_data['text'], self.train_data['label']))}
            self.dev_data = {i:(t,l) for i, (t, l) in enumerate(zip(self.dev_data['text'], self.dev_data['label']))}
            self.id2label = {0: 'world', 1: 'sports', 2:'business', 3:'sci/tech'}
            self.instruction = ""
            self.args.single = True
        ## multichoice QA
        elif args.data=="commonsense_qa":
            data = load_dataset("tau/commonsense_qa")
            self.train_data = data['train']
            self.dev_data = data['validation']
            reform_answer = {"A": 0, "B": 1, "C":2, "D":3, "E":4}
            self.train_data = {i:([t1]+t2['text'],reform_answer[l]) for i, (t1, t2, l) in enumerate(zip(self.train_data['question'], self.train_data['choices'], self.train_data['answerKey']))}
            self.dev_data = {i:([t1]+t2['text'],reform_answer[l]) for i, (t1, t2, l) in enumerate(zip(self.dev_data['question'], self.dev_data['choices'], self.dev_data['answerKey']))}
            self.id2label = {0:"A", 1:"B", 2:"C", 3:"D", 4:"E"}
            self.instruction = ""
            self.args.single = False

        self.label2id = {v:k for k,v in self.id2label.items()}
        self.id2verb = list(self.label2id.keys())
        print(f"load {args.data} datasets")

        if self.args.data == "sst2" or self.args.data == "cr":
            self.template=self.template_binary_sentiment
        elif self.args.data == "trec":
            self.template=self.template_trec
        elif self.args.data == "rte" or self.args.data == "wnli" or self.args.data == "mnli" or self.args.data == "snli":
            self.template = self.template_nli
        elif self.args.data =="sst5":
            self.template = self.template_sst5
        elif self.args.data=="agnews":
            self.template = self.template_agnews
        elif self.args.data=="TweetEval_Hate" or self.args.data=="hate_speech18":
            self.template = self.template_hate
        elif self.args.data=="mrpc":
            self.template = self.template_mrpc
        elif self.args.data=="mqp":
            self.template = self.template_mqp
        elif self.args.data=="commonsense_qa":
            self.template = self.template_cqa

        if self.args.max_train_size == -1:
            print(f"Initial train dataset size :{len(self.train_data.keys())}")
        elif self.args.max_train_size<len(self.train_data.keys()):
            org_train_len = len(self.train_data.keys())
            self.train_data = self.limit_dataset_size(self.train_data, self.args.max_train_size)
            print(f"Initial train dataset size : {org_train_len} -> {len(self.train_data.keys())}")
        else:
            print(f"Initial train dataset size :{len(self.train_data.keys())}")
        
        if self.args.max_dev_size == -1:
            print(f"Initial dev dataset size :{len(self.dev_data.keys())}")
        elif self.args.max_dev_size<len(self.dev_data.keys()):
            org_dev_len = len(self.dev_data.keys())
            self.dev_data = self.limit_dataset_size(self.dev_data, self.args.max_dev_size, False)
            print(f"Initial dev dataset size : {org_dev_len} -> {len(self.dev_data.keys())}")
        else:
            print(f"Initial dev dataset size :{len(self.dev_data.keys())}")

    def split_dataset(self, args, d_name, test_size=0.2):
        if os.path.isdir(f'./dataset_split/{args.data}'):
            data = load_from_disk(f'./dataset_split/{args.data}')
        else:
            os.makedirs(f'./dataset_split', exist_ok=True)
            org_data = load_dataset(d_name)['train']
            idx = [i for i in range(len(org_data))]
            org_data = org_data.add_column("idx", idx)
            if args.data == "hate_speech18":
                org_data = org_data.filter(lambda example: example['label']!=2)
                org_data = org_data.filter(lambda example: example['label']!=3)
            data = org_data.train_test_split(test_size, stratify_by_column="label")
            data.save_to_disk(f'./dataset_split/{args.data}')
        return data
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
    # def template_nli(self, sentence, label=None, mode='train'):
    #     if mode == 'train':
    #         return f"{sentence[0]}\nquestion:{sentence[1]}\ntrue or false?\nanswer:"+f"{self.id2label[label]}\n"
    #     else:
    #         return f"{sentence[0]}\nquestion:{sentence[1]}\ntrue or false?\nanswer:"
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
    def template_mrpc(self, sentence, label=None, n_filler = 0, mode='train'):
        if mode == 'train':
            return f"Sentence 1:{sentence[0]}\nSentence 2:{sentence[1]}\nEquivalent?:" + "..."* n_filler +f"{self.id2label[label]}\n"
        else:
            return f"Sentence 1:{sentence[0]}\nSentence 2:{sentence[1]}\nEquivalent?:" + "..."* n_filler
    def template_mqp(self, sentence, label=None, n_filler = 0, mode='train'):
        if mode == 'train':
            return f"Question 1:{sentence[0]}\nQuestion 2:{sentence[1]}\nSimilar?:" + "..."* n_filler +f"{self.id2label[label]}\n"
        else:
            return f"Question 1:{sentence[0]}\nQuestion 2:{sentence[1]}\nSimilar?:" + "..."* n_filler
    def template_cqa(self, sentence, label=None, n_filler = 0, mode='train'):
        if mode == 'train':
            return f"Question:{sentence[0]}\nChoose answer from choices\nA:{sentence[1]}\nB:{sentence[2]}\nC:{sentence[3]}\nD:{sentence[4]}\nE:{sentence[5]}\nAnswer:{self.id2label[label]}\n"
        else:
            return f"Question:{sentence[0]}\nChoose answer from choices\nA:{sentence[1]}\nB:{sentence[2]}\nC:{sentence[3]}\nD:{sentence[4]}\nE:{sentence[5]}\nAnswer:"

    def __len__(self):
        return len(self.dev_data.keys()) 
    
    def __getitem__(self, idx):
        i, (t, l) = list(self.dev_data.items())[idx]
        prompt = ''
        prompt+=self.template(t, mode = 'inference')
        label = l
        return i, prompt, label