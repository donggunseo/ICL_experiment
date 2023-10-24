import torch
from torch.utils.data import Dataset
import random
from datasets import load_dataset
from tqdm import tqdm
from multiprocessing import Pool
import time
import parmap
import json
import pickle

class ICL_dataset(Dataset):
    def __init__(self, args):
        self.args = args
        if args.data=="CR":
            with open("../data/CR_Train_fixed.json", 'r') as f:
                self.train_data = json.load(f)
            with open("../data/CR_Test_fixed.json", 'r') as f:
                self.dev_data = json.load(f)
            self.id2label = {0: 'negative', 1: 'positive'}
            self.label2id = {'negative': 0, 'positive': 1}
            self.id2verb = ['negative', 'positive']
        elif args.data=="sst2":
            data = load_dataset(args.data)
            self.train_data = data['train']
            self.dev_data = data['validation']
            self.id2label = {0: 'negative', 1: 'positive'}
            self.label2id = {'negative': 0, 'positive': 1}
            self.id2verb = ['negative', 'positive']
        elif args.data=="trec":
            data = load_dataset(args.data)
            self.train_data = data['train']
            self.dev_data = data['test']
            self.train_data = {'sentence' : self.train_data['text'], 'label' : self.train_data['coarse_label']}
            self.dev_data = {'sentence' : self.dev_data['text'], 'label' : self.dev_data['coarse_label']}
            self.id2label = {0:'expression', 1:'entity', 2:'description', 3:'human', 4:'location', 5:'number'}
            self.label2id = {v:k for k,v in self.id2label.items()}
            self.id2verb = list(self.label2id.keys())
        self.train_label = self.train_data['label']
        
        if not self.args.confidence:
            print("sample demonstration randomly")
            self.subsample(args.n_shot)
        else:
            option = "finetune" if self.args.confidence_finetune else "LLM"
            print(f"sample demosntration by confidence score order from {option}")
            self.confidence_sample(args.n_shot)
        
        if self.args.data == "sst2":
            self.template=self.template_sst2
        elif self.args.data == "CR":
            self.template=self.template_CR
        elif self.args.data == "trec":
            self.template=self.template_trec

    def subsample(self, n_shot):
        data_by_cls = {}
        for i in range(len(self.train_label)):
            if self.train_label[i] not in data_by_cls:
                data_by_cls[self.train_label[i]] = []
            data_by_cls[self.train_label[i]].append(i)
        data_subsample = []
        for cls in data_by_cls.keys():
            data_subsampled_by_cls = random.sample(data_by_cls[cls], min(n_shot, len(data_by_cls[cls])))
            data_subsample.extend(data_subsampled_by_cls)
        random.shuffle(data_subsample)
        print(data_subsample)
        self.train_data = {"sentence" : [self.train_data['sentence'][i] for i in data_subsample], 
                           "label" : [self.train_data['label'][i] for i in data_subsample]}
        
    def confidence_sample(self, n_shot):
        data_by_cls = {}
        for i in range(len(self.train_label)):
            if self.train_label[i] not in data_by_cls:
                data_by_cls[self.train_label[i]] = []
            data_by_cls[self.train_label[i]].append(i)
        with open(self.args.confidence_prob_path, 'rb') as f:
            prob_i = pickle.load(f)
        data_subsample = []
        for cls in data_by_cls.keys():
            data_subsampled_by_cls = []
            for item in prob_i:
                if item[0] in data_by_cls[cls]:
                    data_subsampled_by_cls.append(item[0])
                else:
                    continue
                if len(data_subsampled_by_cls)==n_shot:
                    break
            data_subsample.extend(data_subsampled_by_cls)
        random.shuffle(data_subsample)
        print(data_subsample)
        self.train_data = {"sentence" : [self.train_data['sentence'][i] for i in data_subsample], 
                           "label" : [self.train_data['label'][i] for i in data_subsample]}

    def template_sst2(self, sentence, label=None, mode='train'):
        if mode == 'train':
            return f"Review: {sentence}\nSentiment: {self.id2label[label]}\n"
        else:
            return f"Review: {sentence}\nSentiment:"
    def template_CR(self, sentence, label=None, mode='train'):
        if mode == 'train':
            return f"Review: {sentence}\nSentiment: {self.id2label[label]}\n"
        else:
            return f"Review: {sentence}\nSentiment:"
    def template_trec(self, sentence, label=None, mode='train'):
        if mode == 'train':
            return f"Question: {sentence}\nType: {self.id2label[label]}\n"
        else:
            return f"Question: {sentence}\nType:"

    def __len__(self):
        return len(self.dev_data['sentence'])
    
    def __getitem__(self, idx):
        prompt = ''
        for s,l in zip(self.train_data['sentence'], self.train_data['label']):
            prompt+=self.template(s,l, mode='train')
            prompt += '\n'
        
        prompt+=self.template(self.dev_data['sentence'][idx], mode = 'inference')
        dev_label = self.dev_data['label'][idx]
        
        return (prompt, dev_label)



