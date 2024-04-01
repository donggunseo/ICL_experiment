from torch.utils.data import Dataset, DataLoader
import random
from datasets import load_dataset
from tqdm import tqdm
import pickle
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoConfig, AutoTokenizer
import argparse
from utils import set_seed
import torch
import os
import numpy as np
import torch.nn as nn
from itertools import combinations, product, permutations

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
            self.id2label = {0:'expression', 1:'entity', 2:'description', 3:'human', 4:'location', 5:'number'}
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
        self.train_data_by_cls = {}
        for i in range(len(self.train_label)):
            if self.train_label[i] not in self.train_data_by_cls:
                self.train_data_by_cls[self.train_label[i]] = []
            self.train_data_by_cls[self.train_label[i]].append(i)
        self.dev_data_by_cls = {}
        for i in range(len(self.dev_label)):
            if self.dev_label[i] not in self.dev_data_by_cls:
                self.dev_data_by_cls[self.dev_label[i]] = []
            self.dev_data_by_cls[self.dev_label[i]].append(i)              

        
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

    def exhaustive_search(self):
        data_by_cls_train = {}
        for i in range(len(self.train_label)):
            if self.train_label[i] not in data_by_cls_train:
                data_by_cls_train[self.train_label[i]] = []
            data_by_cls_train[self.train_label[i]].append(i)
        for cls in data_by_cls_train.keys():
            subset_train = random.sample(data_by_cls_train[cls], 4)
            data_by_cls_train[cls] = subset_train
        
        data_by_cls_dev = {}
        for i in range(len(self.dev_label)):
            if self.dev_label[i] not in data_by_cls_dev:
                data_by_cls_dev[self.dev_label[i]] = []
            data_by_cls_dev[self.dev_label[i]].append(i)
        dev_data_subsample = []
        for cls in data_by_cls_dev.keys():
            subset_dev = random.sample(data_by_cls_dev[cls], 5)
            dev_data_subsample.extend(subset_dev)
        self.dev_data = {'sentence': [self.dev_sentence[i] for i in dev_data_subsample], "label": [self.dev_label[i] for i in dev_data_subsample]}

        t = []
        for cls in data_by_cls_train.keys():
            temp = list(combinations(data_by_cls_train[cls],2))
            t.append(temp)
        t = list(product(*t))
        ## (element ((1,2),(5,6))
        t = [[l[0][0],l[0][1], l[1][0], l[1][1]] for l in t]

        self.demonstration_data_org = []
        self.demonstration_data_ori = []
        for t1 in t:
            t1 = list(permutations(t1,4))
            t1 = [list(l) for l in t1]
            self.demonstration_data_org.extend(t1)
            for res in t1:
                org = {"sentence" : [self.train_sentence[i] for i in res], "label" : [self.train_label[i] for i in res]}
                self.demonstration_data_ori.append(org)

        

    
    def random_sample(self, n_shot):
        data_by_cls = {}
        for i in range(len(self.train_label)):
            if self.train_label[i] not in data_by_cls:
                data_by_cls[self.train_label[i]] = []
            data_by_cls[self.train_label[i]].append(i)
        self.demonstration_data_ori = []
        for _ in range(self.num_org):
            data_subsample = []
            for cls in data_by_cls.keys():
                data_subsampled_by_cls = random.sample(data_by_cls[cls], n_shot)
                data_subsample.extend(data_subsampled_by_cls)
            random.shuffle(data_subsample)
            org = {"sentence" : [self.train_sentence[i] for i in data_subsample], "label" : [self.train_label[i] for i in data_subsample]}
            self.demonstration_data_ori.append(org)
    def random_sample_v2(self,n_shot):
        data_by_cls = {}
        for i in range(len(self.train_label)):
            if self.train_label[i] not in data_by_cls:
                data_by_cls[self.train_label[i]] = []
            data_by_cls[self.train_label[i]].append(i)
        self.demonstration_data_ori = []
        for _ in range(self.num_org):
            data_subsample_v2 = []
            for cls in data_by_cls.keys():
                data_subsampled_by_cls_v2 = random.sample(data_by_cls[cls], 1)
                data_subsample_v2.extend(data_subsampled_by_cls_v2)
            random.shuffle(data_subsample_v2)
            data_subsample = []
            for cls in data_by_cls.keys():
                data_subsampled_by_cls = random.sample(data_by_cls[cls], n_shot-1)
                data_subsample.extend(data_subsampled_by_cls)
            random.shuffle(data_subsample)
            data_subsample = data_subsample_v2+data_subsample
            org = {"sentence" : [self.train_sentence[i] for i in data_subsample], "label" : [self.train_label[i] for i in data_subsample]}
            self.demonstration_data_ori.append(org)
    

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
        return len(self.dev_data['sentence']) 
    
    def __getitem__(self, idx):
        prompt = ''
        prompt+=self.template(self.dev_data['sentence'][idx], mode = 'inference')
        label = self.dev_data['label'][idx]
        return (prompt, label)

def parse_args():
    parser = argparse.ArgumentParser(description="In-Context Learning baseline.")
    parser.add_argument("--llm", type=str, default='llama2_7b')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="sst2")
    parser.add_argument("--n_shot", type=int, default=4)
    parser.add_argument("--n_org", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    return args

model_dict ={
    "llama2_7b" : "meta-llama/Llama-2-7b-hf",
    "llama2_13b" : "meta-llama/Llama-2-13b-hf",
    "opt_6.7b" : "facebook/opt-6.7b",
    "gpt2_xl" : "gpt2-xl"
}

def main():
    args = parse_args()
    set_seed(args)
    dataset = ICL_dataset(args)
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle=False)
    id2verb = dataset.id2verb
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dict[args.llm])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model_config = AutoConfig.from_pretrained(model_dict[args.llm])
    model = AutoModelForCausalLM.from_pretrained(model_dict[args.llm], config = model_config)
    model.to(device)
    model.eval()


    demon_set = dataset.demonstration_data_ori
    demon_set_idx = dataset.demonstration_data_org
    template = dataset.template
    res = []
    for d_idx, d in tqdm(zip(demon_set_idx, demon_set)):
        prompt = ''
        cnt = 0
        for s, l in zip(d['sentence'], d['label']):
            prompt+=template(s,l, mode='train')
            prompt += '\n'
        for batch in dataloader:
            prompt2 = prompt + batch[0][0]
            label = batch[1][0].item()
            inputs = tokenizer(prompt2, return_tensors="pt", padding=True).to(device=model.device)
            with torch.no_grad():
                logits = model.forward(input_ids=inputs['input_ids'],
                                        attention_mask=inputs['attention_mask'],
                                        return_dict=True).logits.detach().cpu()
            gen_logits = logits[:,-1,:]
            prob_per_cls = []
            for label_verb in id2verb:
                label_verb_token_id = tokenizer.encode(label_verb, add_special_tokens=False)[0]
                prob_per_cls.append(gen_logits[:, label_verb_token_id].item())
            prediction = np.argmax(np.array(prob_per_cls))
            if label==prediction:
                cnt+=1
        res.append([d_idx,cnt/10])
    with open('./res_CR.pickle', 'wb') as f:
        pickle.dump(res, f)
    
    uniq_d_idx = []
    for l in res:
        uniq_d_idx.extend(l[0])
    uniq_d_idx = list(set(uniq_d_idx))

    inf_d_idx = {}
    for uniq in uniq_d_idx:
        avg_acc = []
        for l in res:
            if uniq in l[0]:
                avg_acc.append(l[1])
        avg = sum(avg_acc)/len(avg_acc)
        inf_d_idx[uniq] = avg
    inf_d_idx = {k:v for k,v in sorted(inf_d_idx.items(), key=lambda item: -item[1])}

    with open('./inf_d_idx_CR.pickle', 'wb') as f:
        pickle.dump(inf_d_idx,f)

    

    # inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device=model.device)
    # with torch.no_grad():
    #     logits = model.forward(input_ids=inputs['input_ids'],
    #                             attention_mask=inputs['attention_mask'],
    #                             return_dict=True).logits.detach().cpu()
    # input_ids = inputs['input_ids'][0].detach().cpu()
    # label_words_pred_idx = []
    # for i in range(len(input_ids)-1):
    #     if input_ids[i]==13 and input_ids[i+1]==13:
    #         label_words_pred_idx.append(i-2)
    # logits = logits.squeeze(0)
    # gen_logits = torch.index_select(logits, 0, torch.tensor(label_words_pred_idx))
    # prob_per_cls = []
    # for label_verb in id2verb:
    #     label_verb_token_id = tokenizer.encode(label_verb, add_special_tokens=False)[0]
    #     prob_per_cls.append(gen_logits[:, label_verb_token_id])
    # prob = torch.stack(prob_per_cls, dim=1)
    # label = torch.tensor(demon['label'])
    # loss = criterion(prob,label)
    # label_text = [id2verb[l] for l in label.tolist()]
    # for g, lt in zip(gen_logits, label_text):
    #     g = torch.softmax(g, dim=-1)
    #     val, idx = torch.topk(g,8)
    #     print(lt)
    #     pred_for_each_pos = [(tokenizer.decode(i, skip_special_tokens=True),round(v,4)) for i,v in zip(idx.tolist(),val.tolist())]
    #     print(pred_for_each_pos)
    #     label_verb_token_ids = [tokenizer.encode(l, add_special_tokens=False)[0] for l in id2verb]
    #     print(round(sum([g[i].item() for i in label_verb_token_ids]),4))
    #     print("---------------------------------------")

if __name__ == "__main__":
    main()
