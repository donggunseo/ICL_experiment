from torch.utils.data import Dataset
import random
from datasets import load_dataset
from tqdm import tqdm
import json
import pickle

class ICL_dataset(Dataset):
    def __init__(self, args):
        self.args = args
        if args.data=="CR":
            data = load_dataset("SetFit/CR")
            self.train_data = data['train']
            self.dev_data = data['test']
            self.train_data = {'sentence' : self.train_data['text'], 'label' : self.train_data['label']}
            self.dev_data = {'sentence' : self.dev_data['text'], 'label' : self.dev_data['label']}
            self.id2label = {0: 'negative', 1: 'positive'}
            self.label2id = {'negative': 0, 'positive': 1}
            self.id2verb = ['negative', 'positive']
        elif args.data=="sst2":
            data = load_dataset(args.data)
            self.train_data = data['train']
            self.dev_data = data['validation']
            self.train_data = {'sentence' : self.train_data['sentence'], 'label' : self.train_data['label']}
            self.dev_data = {'sentence' : self.dev_data['sentence'], 'label' : self.dev_data['label']}
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
        elif args.data=='subj':
            data = load_dataset("SetFit/subj")
            self.train_data = data['train']
            self.dev_data = data['test']
            self.train_data = {'sentence' : self.train_data['text'], 'label' : self.train_data['label']}
            self.dev_data = {'sentence' : self.dev_data['text'], 'label' : self.dev_data['label']}
            self.id2label = {0:'objective', 1:'subjective'}
            self.label2id = {v:k for k,v in self.id2label.items()}
            self.id2verb = list(self.label2id.keys())
        elif args.data=='rte':
            data = load_dataset("glue", 'rte')
            self.train_data = data['train']
            self.dev_data = data['validation']
            sentence1 = self.train_data['sentence1']
            sentence2 = self.train_data['sentence2']
            concat_s = [[s1, s2] for s1, s2 in zip(sentence1, sentence2)]
            self.train_data = {'sentence' : concat_s, 'label' : self.train_data['label']}
            sentence1 = self.dev_data['sentence1']
            sentence2 = self.dev_data['sentence2']
            concat_s = [[s1, s2] for s1, s2 in zip(sentence1, sentence2)]
            self.dev_data = {'sentence' : concat_s, 'label' : self.dev_data['label']}
            self.id2label = {0:'true', 1:'false'}
            self.label2id = {v:k for k,v in self.id2label.items()}
            self.id2verb = list(self.label2id.keys())
        elif args.data=='sst5':
            data = load_dataset("SetFit/sst5")
            self.train_data = data['train']
            self.dev_data = data['test']
            self.train_data = {'sentence' : self.train_data['text'], 'label' : self.train_data['label']}
            self.dev_data = {'sentence' : self.dev_data['text'], 'label' : self.dev_data['label']}
            self.id2label = {0: 'terrible', 1: 'bad', 2:'okay', 3:'good', 4:'great'}
            self.label2id = {v:k for k,v in self.id2label.items()}
            self.id2verb = list(self.label2id.keys())
        elif args.data=="agnews":
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
        if self.args.selection == "whole_random":
            print("sample demonstration randomly without stratify")
            self.whole_random_sample(args.n_shot)
        elif self.args.selection == "random":
            print("sample demonstration randomly")
            self.random_sample(args.n_shot)
        elif self.args.selection == "finetune_confidence" or self.args.selection == "LLM_confidence":
            print(f"sample demosntration by {self.args.selection} score order")
            self.confidence_sample(args.n_shot, mode=self.args.selection)
        elif self.args.selection == "knn":
            print("sample demonstration by knn")
            self.knn_sample(args.n_shot)
        elif self.args.selection == "knn_stratify":
            print("sample demonstration by knn_stratify")
            self.knn_stratify_sample(args.n_shot)
        
        if self.args.data == "sst2":
            self.template=self.template_sst2
        elif self.args.data == "CR":
            self.template=self.template_CR
        elif self.args.data == "trec":
            self.template=self.template_trec
        elif self.args.data == "subj":
            self.template = self.template_subj
        elif self.args.data == "rte":
            self.template = self.template_rte
        elif self.args.data =="sst5":
            self.template = self.template_sst5
        elif self.args.data=="agnews":
            self.template = self.template_agnews
    
    def whole_random_sample(self, n_shot):
        idx = [i for i in range(len(self.train_label))]
        data_subsample = random.sample(idx, min(n_shot*len(self.id2verb), len(idx)))
        random.shuffle(data_subsample)
        print(data_subsample)
        self.demonstration_data = {"sentence" : [self.train_sentence[i] for i in data_subsample], 
                           "label" : [self.train_label[i] for i in data_subsample]}
        
    def random_sample(self, n_shot):
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
        self.demonstration_data = {"sentence" : [self.train_sentence[i] for i in data_subsample], 
                           "label" : [self.train_label[i] for i in data_subsample]}
        
    def confidence_sample(self, n_shot, mode):
        data_by_cls = {}
        for i in range(len(self.train_label)):
            if self.train_label[i] not in data_by_cls:
                data_by_cls[self.train_label[i]] = []
            data_by_cls[self.train_label[i]].append(i)
        if mode == "finetune_confidence":
            with open(f"./Finetune_confidence/{self.args.data}_finetune.json", 'r') as f:
                confidence = json.load(f)
        elif mode == "LLM_confidence":
            with open(f"./LLM_confidence/{self.args.llm}/{self.args.data}_finetune.json", 'r') as f:
                confidence = json.load(f)
        confidence = sorted(confidence.items(), key = lambda item: -item[1]['confidence_score'])
        data_subsample = []
        for cls in data_by_cls.keys():
            data_subsampled_by_cls = []
            for item in confidence:
                if int(item[0]) in data_by_cls[cls]:
                    data_subsampled_by_cls.append(int(item[0]))
                else:
                    continue
                if len(data_subsampled_by_cls)==n_shot:
                    break
            data_subsample.extend(data_subsampled_by_cls)
        random.shuffle(data_subsample)
        print(data_subsample)
        self.demonstration_data = {"sentence" : [self.train_data['sentence'][i] for i in data_subsample], 
                           "label" : [self.train_data['label'][i] for i in data_subsample]}
    
    def knn_sample(self, n_shot):
        with open(f'./SBERT_emb/{self.args.data}_train_dev_knn_indices.pickle', 'rb') as f:
            knn_indices = pickle.load(f)
        self.demonstration_data = []
        k = n_shot * len(self.id2verb) ## we do not consider stratify class, so add n_shot * n_classes for total demonstrations
        for i in tqdm(range(len(self.dev_data['label']))):
            knn_indice = knn_indices[i][:k]
            random.shuffle(knn_indice)
            self.demonstration_data.append({"sentence" : [self.train_sentence[i] for i in knn_indice], "label" : [self.train_label[i] for i in knn_indice]})

    def knn_stratify_sample(self, n_shot):
        with open(f'./SBERT_emb/{self.args.data}_train_dev_knn_indices.pickle', 'rb') as f:
            knn_indices = pickle.load(f)
        data_by_cls = {}
        for i in range(len(self.train_label)):
            if self.train_label[i] not in data_by_cls:
                data_by_cls[self.train_label[i]] = []
            data_by_cls[self.train_label[i]].append(i)
        self.demonstration_data = []
        for i in tqdm(range(len(self.dev_data['label']))):
            data_subsample=[]
            knn_indice = knn_indices[i]
            for cls in data_by_cls.keys():
                data_subsampled_by_cls = []
                for idx in knn_indice:
                    if idx in data_by_cls[cls]:
                        data_subsampled_by_cls.append(idx)
                    else:
                        continue
                    if len(data_subsampled_by_cls)==n_shot:
                        break
                data_subsample.extend(data_subsampled_by_cls)
            random.shuffle(data_subsample)
            self.demonstration_data.append({"sentence" : [self.train_sentence[i] for i in data_subsample], "label" : [self.train_label[i] for i in data_subsample]})


    def template_sst2(self, sentence, label=None, mode='train'):
        if mode == 'train':
            return f"Review: {sentence}\nSentiment: {self.id2label[label]}\n"
        else:
            return f"Review: {sentence}\nSentiment: "
    def template_CR(self, sentence, label=None, mode='train'):
        if mode == 'train':
            return f"Review: {sentence}\nSentiment: {self.id2label[label]}\n"
        else:
            return f"Review: {sentence}\nSentiment: "
    def template_trec(self, sentence, label=None, mode='train'):
        if mode == 'train':
            return f"Question: {sentence}\nType: {self.id2label[label]}\n"
        else:
            return f"Question: {sentence}\nType: "
    def template_subj(self, sentence, label=None, mode='train'):
        if mode == 'train':
            return f"Input: {sentence}\nType: {self.id2label[label]}\n"
        else:
            return f"Input: {sentence}\nType: "
    def template_rte(self, sentence, label=None, mode='train'):
        if mode == 'train':
            return f"Premise: {sentence[0]}\nHypothesis: {sentence[1]}\nPredicton: {self.id2label[label]}\n"
        else:
            return f"Premise: {sentence[0]}\nHypothesis: {sentence[1]}\nPredicton: "
    def template_sst5(self, sentence, label=None, mode='train'):
        if mode == 'train':
            return f"Review: {sentence}\nSentiment: {self.id2label[label]}\n"
        else:
            return f"Review: {sentence}\nSentiment: "
    def template_agnews(self, sentence, label=None, mode='train'):
        if mode == 'train':
            return f"Input: {sentence}\nTopic: {self.id2label[label]}\n"
        else:
            return f"Input: {sentence}\nTopic: "

    def __len__(self):
        return len(self.dev_data['sentence'])
    
    def __getitem__(self, idx):
        prompt = ''
        if self.args.selection == "random" or self.args.selection == "finetune_confidence" or self.args.selection == "LLM_confidence" or self.args.selection == "whole_random":
            for s, l in zip(self.demonstration_data['sentence'], self.demonstration_data['label']):
                prompt+=self.template(s,l, mode='train')
                prompt += '\n'
            prompt+=self.template(self.dev_data['sentence'][idx], mode = 'inference')
            label = self.dev_data['label'][idx]
            return (prompt, label)
        elif self.args.selection == "knn" or self.args.selection == "knn_stratify":
            for s, l in zip(self.demonstration_data[idx]['sentence'], self.demonstration_data[idx]['label']):
                prompt+=self.template(s,l, mode='train')
                prompt += '\n'
            prompt+=self.template(self.dev_data['sentence'][idx], mode = 'inference')
            label = self.dev_data['label'][idx]
            return (prompt, label)



