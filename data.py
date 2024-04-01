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
            self.train_data = {'sentence' : self.train_data['text'], 'label' : self.train_data['label']}
            self.dev_data = {'sentence' : self.dev_data['text'], 'label' : self.dev_data['label']}
            self.id2label = {0: 'negative', 1: 'positive'}
        elif args.data=="sst2": ## 영화 리뷰 감정 분류 데이터셋 train 67349 validation 872 test 1821(X) 
            data = load_dataset('glue', 'sst2')
            self.train_data = data['train']
            self.dev_data = data['validation']
            self.train_data = {'sentence' : self.train_data['sentence'], 'label' : self.train_data['label']}
            self.dev_data = {'sentence' : self.dev_data['sentence'], 'label' : self.dev_data['label']}
            self.id2label = {0: 'negative', 1: 'positive'}
        ## sentiment classification (polarity)
        elif args.data=='sst5': ## 영화 리뷰 감정 분류 데이터셋 train 8544 validation 1101 test 2210 
            data = load_dataset("SetFit/sst5")
            self.train_data = data['train']
            self.dev_data = data['test']
            self.train_data = {'sentence' : self.train_data['text'], 'label' : self.train_data['label']}
            self.dev_data = {'sentence' : self.dev_data['text'], 'label' : self.dev_data['label']}
            self.id2label = {0: 'terrible', 1: 'bad', 2:'okay', 3:'good', 4:'great'}
        ## NLI Classification  
        elif args.data=='rte': ## NLI 형 데이터셋 train 2490 validation 277 test 3000(X)
            data = load_dataset("glue", 'rte')
            self.train_data = data['train']
            self.dev_data = data['validation']
            self.train_data = {'sentence' : [[s1, s2] for s1, s2 in zip(self.train_data['sentence1'], self.train_data['sentence2'])], 'label' : self.train_data['label']}
            self.dev_data = {'sentence' : [[s1, s2] for s1, s2 in zip(self.dev_data['sentence1'], self.dev_data['sentence2'])], 'label' : self.dev_data['label']}
            self.id2label = {0:'true', 1:'false'}
        elif args.data=='wnli':
            data = load_dataset("glue", 'wnli') ## NLI 형 데이터셋 train 635 validation 71 test 146(x)
            self.train_data = data['train']
            self.dev_data = data['validation']
            self.train_data = {'sentence' : [[s1, s2] for s1, s2 in zip(self.train_data['sentence1'], self.train_data['sentence2'])], 'label' : self.train_data['label']}
            self.dev_data = {'sentence' : [[s1, s2] for s1, s2 in zip(self.dev_data['sentence1'], self.dev_data['sentence2'])], 'label' : self.dev_data['label']}
            self.id2label = {0:'false', 1:'true'}
        ## Question type Classification
        elif args.data=="trec": ## 질문 유형 분류 데이터셋 train 5452 test 500
            data = load_dataset('trec')
            self.train_data = data['train']
            self.dev_data = data['test']
            self.train_data = {'sentence' : self.train_data['text'], 'label' : self.train_data['coarse_label']}
            self.dev_data = {'sentence' : self.dev_data['text'], 'label' : self.dev_data['coarse_label']}
            self.id2label = {0:'abbreviation', 1:'entity', 2:'description', 3:'human', 4:'location', 5:'number'}
        ## Topic Classification
        elif args.data=="agnews": ## 기사 주제 분류 데이터셋 train 120000 test 7600
            data = load_dataset("ag_news")
            self.train_data = data['train']
            self.dev_data = data['test']
            self.train_data = {'sentence' : self.train_data['text'], 'label' : self.train_data['label']}
            self.dev_data = {'sentence' : self.dev_data['text'], 'label' : self.dev_data['label']}
            self.id2label = {0: 'world', 1: 'sports', 2:'business', 3:'technology'}
        self.label2id = {v:k for k,v in self.id2label.items()}
        self.id2verb = list(self.label2id.keys())
        self.train_label = self.train_data['label']
        self.train_sentence = self.train_data['sentence']
        self.dev_label = self.dev_data['label']
        self.dev_sentence = self.dev_data['sentence']
        self.train_data_by_cls = self.split_by_class(self.train_label)
        self.dev_data_by_cls = self.split_by_class(self.dev_label)
        print(f"load {args.data} datasets")

        if self.args.data == "sst2":
            self.template=self.template_sst2
        elif self.args.data == "cr":
            self.template=self.template_cr
        elif self.args.data == "trec":
            self.template=self.template_trec
        elif self.args.data == "rte" or self.args.data == "wnli":
            self.template = self.template_nli_2
        elif self.args.data =="sst5":
            self.template = self.template_sst5
        elif self.args.data=="agnews":
            self.template = self.template_agnews

        if self.args.max_train_size<len(self.train_sentence):
            self.train_data_by_cls = self.limit_dataset_size(self.train_data_by_cls, self.args.max_train_size)
        print(f"Initial train dataset size : {len(self.train_sentence)} -> {sum([len(v) for v in self.train_data_by_cls.values()])}")
        
        if self.args.max_dev_size<len(self.dev_sentence):
            if os.path.isfile(f'./limited_devset_idx/{self.args.data}_{self.args.max_dev_size}.json'):
                with open(f'./limited_devset_idx/{self.args.data}_{self.args.max_dev_size}.json', 'r') as f:
                    self.dev_data_by_cls = json.load(f)
            else:
                self.dev_data_by_cls = self.limit_dataset_size(self.dev_data_by_cls, self.args.max_dev_size, True)
            total_dev_data_idx = []
            for v in self.dev_data_by_cls.values():
                total_dev_data_idx.extend(v)
            self.dev_sentence = [self.dev_sentence[i] for i in total_dev_data_idx]
            self.dev_label = [self.dev_label[i] for i in total_dev_data_idx]
        print(f"Initial dev dataset size : {len(self.dev_sentence)}")


    def split_by_class(self, labels):
        data_by_cls = {}
        for i in range(len(labels)):
            if labels[i] not in data_by_cls:
                data_by_cls[labels[i]]=[]
            data_by_cls[labels[i]].append(i)
        return data_by_cls
    ## only when limited size is smaller than total dataset size
    def limit_dataset_size(self, data_by_cls, dataset_size, dev=False):
        num_data_per_label = [len(v) for v in data_by_cls.values()]
        total = sum(num_data_per_label)
        num_data_per_label = [(l * dataset_size)//total for l in num_data_per_label]
        j=0
        for cls in data_by_cls.keys():
            data_by_cls[cls] = random.sample(data_by_cls[cls], num_data_per_label[j])
            j+=1
        if dev:
            with open(f'./limited_devset_idx/{self.args.data}_{dataset_size}.json', 'w') as f:
                json.dump(data_by_cls, f)
        return data_by_cls
    
    def template_sst2(self, sentence, label=None, mode='train'):
        if mode == 'train':
            return f"Review: {sentence}\nSentiment: {self.id2label[label]}\n"
        else:
            return f"Review: {sentence}\nSentiment: "
    def template_cr(self, sentence, label=None, mode='train'):
        if mode == 'train':
            return f"Review: {sentence}\nSentiment: {self.id2label[label]}\n"
        else:
            return f"Review: {sentence}\nSentiment: "
    def template_trec(self, sentence, label=None, mode='train'):
        if mode == 'train':
            return f"Question: {sentence}\nType: {self.id2label[label]}\n"
        else:
            return f"Question: {sentence}\nType: "
    def template_nli_3(self, sentence, label=None, mode='train'):
        if mode == 'train':
            return f"{sentence[0]}\nquestion: {sentence[1]} true, false or neutral?\nAnswer: {self.id2label[label]}\n"
        else:
            return f"{sentence[0]}\nquestion: {sentence[1]} true, false or neutral?\nAnswer: "
    def template_nli_2(self, sentence, label=None, mode='train'):
        if mode == 'train':
            return f"{sentence[0]}\nquestion: {sentence[1]} true or false?\nAnswer: {self.id2label[label]}\n"
        else:
            return f"{sentence[0]}\nquestion: {sentence[1]} true or false?\nAnswer: "
    def template_sst5(self, sentence, label=None, mode='train'):
        if mode == 'train':
            return f"Review: {sentence}\nSentiment: {self.id2label[label]}\n"
        else:
            return f"Review: {sentence}\nSentiment: "
    def template_agnews(self, sentence, label=None, mode='train'):
        if mode == 'train':
            return f"Article: {sentence}\nTopic: {self.id2label[label]}\n"
        else:
            return f"Article: {sentence}\nTopic: "

    def __len__(self):
        return len(self.dev_sentence) 
    
    def __getitem__(self, idx):
        prompt = ''
        prompt+=self.template(self.dev_sentence[idx], mode = 'inference')
        label = self.dev_label[idx]
        return prompt, label