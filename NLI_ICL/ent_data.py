from torch.utils.data import Dataset
import random
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import json
import pickle
import os

class entailment_dataset(Dataset):
    def __init__(self, args):
        self.args = args
        if args.data=="rte":
            data = load_dataset("glue", 'rte')
            self.train_data = data['train']
            self.dev_data = data['validation']
            self.train_data = {i:([t1,t2],l) for i, (t1, t2, l) in enumerate(zip(self.train_data['sentence1'], self.train_data['sentence2'], self.train_data['label']))}
            self.dev_data =  {i:([t1,t2],l) for i, (t1, t2, l) in enumerate(zip(self.dev_data['sentence1'], self.dev_data['sentence2'], self.dev_data['label']))}
            self.id2label = {0:'yes', 1:'no'}
            self.instruction = 'In this task, you are given two sentences. Indicate if the first sentence clearly entails the second sentence (i.e., one can conclude the 2nd sentence by reading the 1st one). Indicate your answer with \'yes\' if the first sentence entails the second sentence, otherwise answer with \'no\'.'
        elif args.data=="wnli":
            data = load_dataset("glue", 'wnli') 
            self.train_data = data['train']
            self.dev_data = data['validation']
            self.train_data = {i:([t1,t2],l) for i, (t1, t2, l) in enumerate(zip(self.train_data['sentence1'], self.train_data['sentence2'], self.train_data['label']))}
            self.dev_data =  {i:([t1,t2],l) for i, (t1, t2, l) in enumerate(zip(self.dev_data['sentence1'], self.dev_data['sentence2'], self.dev_data['label']))}
            self.id2label = {0:'no', 1:'yes'}
            self.instruction = 'In this task, you are given two sentences. Indicate if the first sentence clearly entails the second sentence (i.e., one can conclude the 2nd sentence by reading the 1st one). Indicate your answer with \'yes\' if the first sentence entails the second sentence, otherwise answer with \'no\'.'
        elif args.data=="mnli":
            data = load_dataset("glue", 'mnli') 
            self.train_data = data['train']
            self.dev_data = data['validation_matched']
            self.train_data = {i:([t1,t2],l) for i, (t1, t2, l) in enumerate(zip(self.train_data['premise'], self.train_data['hypothesis'], self.train_data['label']))}
            self.dev_data =  {i:([t1,t2],l) for i, (t1, t2, l) in enumerate(zip(self.dev_data['premise'], self.dev_data['hypothesis'], self.dev_data['label']))}
            self.id2label = {0:'yes', 1:'neither', 2:'no'}
            self.instruction = 'In this task, you are given two sentences. Indicate if the first sentence clearly entails the second sentence (i.e., one can conclude the 2nd sentence by reading the 1st one). Indicate your answer with \'yes\' if the first sentence entails the second sentence. If the first sentence contradicts the second sentence, answer with \'no\'. Otherwise, answer with \'neither\'.'
        elif args.data=="snli":
            data = load_dataset('stanfordnlp/snli') 
            self.train_data = data['train']
            self.dev_data = data['test']
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
            self.instruction = 'In this task, you are given two sentences. Indicate if the first sentence is clearly equivalent with the second sentence (i.e., one can conclude the 2nd sentence by reading the 1st one). Indicate your answer with \'yes\' if the first sentence is equivalent with the second sentence, otherwise answer with \'no\'.'
        elif args.data=="mqp":
            data = self.split_dataset(self.args, d_name='medical_questions_pairs', test_size=0.1)
            self.train_data = data['train']
            self.dev_data = data['test']
            self.train_data = {i:([t1,t2],l) for i, t1, t2, l in zip(self.train_data['idx'], self.train_data['question_1'], self.train_data['question_2'], self.train_data['label'])}
            self.dev_data =  {i:([t1,t2],l) for i, t1, t2, l in zip(self.dev_data['idx'], self.dev_data['question_1'], self.dev_data['question_2'], self.dev_data['label'])}
            self.id2label = {0:'no', 1:'yes'}
            self.instruction = 'In this task, you are given two medical questions. Indicate if the first question is clearly similar with the second question (i.e., two questions have the same connotation or meaning). Indicate your answer with \'yes\' if the first question is similar with the second sentence, otherwise answer with \'no\'.'
        
        self.label2id = {v:k for k,v in self.id2label.items()}
        self.id2verb = list(self.label2id.keys())
        print(f"load {args.data} datasets")
        if self.args.data == "rte" or self.args.data == "wnli":
            self.template = self.template_nli
        elif self.args.data=="mrpc":
            self.template = self.template_nli
        elif self.args.data=="mqp":
            self.template = self.template_nli

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

    def template_nli(self, sentence, label=None,  mode='train'):
        if mode == 'train':
            return f"Sentence 1:{sentence[0]}\nSentence 2:{sentence[1]}\nEntailment?:" + f"{self.id2label[label]}\n"
        else:
            return f"Sentence 1:{sentence[0]}\nSentence 2:{sentence[1]}\nEntailment?:"
    def template_mrpc(self, sentence, label=None, mode='train'):
        if mode == 'train':
            return f"Sentence 1:{sentence[0]}\nSentence 2:{sentence[1]}\nEquivalent?:" +f"{self.id2label[label]}\n"
        else:
            return f"Sentence 1:{sentence[0]}\nSentence 2:{sentence[1]}\nEquivalent?:"
    def template_mqp(self, sentence, label=None, mode='train'):
        if mode == 'train':
            return f"Question 1:{sentence[0]}\nQuestion 2:{sentence[1]}\nSimilar?:" +f"{self.id2label[label]}\n"
        else:
            return f"Question 1:{sentence[0]}\nQuestion 2:{sentence[1]}\nSimilar?:"

    def split_dataset(self, args, d_name, test_size=0.1):
        if os.path.isdir(f'./dataset_split/{args.data}'):
            data = load_from_disk(f'./dataset_split/{args.data}')
        else:
            os.makedirs(f'./dataset_split', exist_ok=True)
            org_data = load_dataset(d_name)['train']
            idx = [i for i in range(len(org_data))]
            org_data = org_data.add_column("idx", idx)
            data = org_data.train_test_split(test_size, stratify_by_column="label")
            data.save_to_disk(f'./dataset_split/{args.data}')
    
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
    
    def __len__(self):
        return len(self.dev_data.keys()) 
    
    def __getitem__(self, idx):
        i, (t, l) = list(self.dev_data.items())[idx]
        prompt = ''
        prompt+=self.template(t, mode = 'inference')
        label = l
        return i, prompt, label