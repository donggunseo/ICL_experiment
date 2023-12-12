import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM, AutoConfig
import argparse
from datasets import load_dataset
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import json
from utils import set_seed
import random
import os

def parse_args():
    parser = argparse.ArgumentParser(description="In-Context Learning confidence score experiment")
    parser.add_argument("--LLM_model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="sst2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--random", action="store_true", default=False) ## 키면 random 끄면 KNN
    parser.add_argument("--n_shot", type=int, default=4)
    parser.add_argument("--save_path", type=str, default="llama2_7b")
    args = parser.parse_args()

    return args

class LLM_Confidence_dataset(Dataset):
    def __init__(self, args):
        self.args = args
        if args.data=="CR":
            data = load_dataset("SetFit/CR")
            self.train_data = data['train']
            self.train_data = {'sentence' : self.train_data['text'], 'label' : self.train_data['label']}
            self.id2label = {0: 'negative', 1: 'positive'}
            self.label2id = {'negative': 0, 'positive': 1}
            self.id2verb = ['negative', 'positive']
        elif args.data=="sst2":
            data = load_dataset(args.data)
            self.train_data = data['train']
            self.train_data = {'sentence' : self.train_data['sentence'], 'label' : self.train_data['label']}
            self.id2label = {0: 'negative', 1: 'positive'}
            self.label2id = {'negative': 0, 'positive': 1}
            self.id2verb = ['negative', 'positive']
        elif args.data=="trec":
            data = load_dataset(args.data)
            self.train_data = data['train']
            self.train_data = {'sentence' : self.train_data['text'], 'label' : self.train_data['coarse_label']}
            self.id2label = {0:'expression', 1:'entity', 2:'description', 3:'human', 4:'location', 5:'number'}
            self.label2id = {v:k for k,v in self.id2label.items()}
            self.id2verb = list(self.label2id.keys())
        elif args.data=='subj':
            data = load_dataset("SetFit/subj")
            self.train_data = data['train']
            self.train_data = {'sentence' : self.train_data['text'], 'label' : self.train_data['label']}
            self.id2label = {0:'objective', 1:'subjective'}
            self.label2id = {v:k for k,v in self.id2label.items()}
            self.id2verb = list(self.label2id.keys())
        elif args.data=='rte':
            data = load_dataset("glue", 'rte')
            self.train_data = data['train']
            sentence1 = self.train_data['sentence1']
            sentence2 = self.train_data['sentence2']
            concat_s = [[s1, s2] for s1, s2 in zip(sentence1, sentence2)]
            self.train_data = {'sentence' : concat_s, 'label' : self.train_data['label']}
            self.id2label = {0:'true', 1:'false'}
            self.label2id = {v:k for k,v in self.id2label.items()}
            self.id2verb = list(self.label2id.keys())
        elif args.data=='sst5':
            data = load_dataset("SetFit/sst5")
            self.train_data = data['train']
            self.train_data = {'sentence' : self.train_data['text'], 'label' : self.train_data['label']}
            self.id2label = {0: 'terrible', 1: 'bad', 2:'okay', 3:'good', 4:'great'}
            self.label2id = {v:k for k,v in self.id2label.items()}
            self.id2verb = list(self.label2id.keys())
        elif args.data=="agnews":
            data = load_dataset("ag_news")
            self.train_data = data['train']
            self.train_data = {'sentence' : self.train_data['text'], 'label' : self.train_data['label']}
            self.id2label = {0: 'world', 1: 'sports', 2:'business', 3:'technology'}
            self.label2id = {v:k for k,v in self.id2label.items()}
            self.id2verb = list(self.label2id.keys())

        self.train_label = self.train_data['label']
        self.train_sentence = self.train_data['sentence']
        
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
        
        if self.args.n_shot!=0:
            if self.args.random:
                self.random_sample(self.args.n_shot)
            else:
                self.knn_sample(self.args.n_shot)
        else:
            self.train_data['idx'] = [i for i in range(len(self.train_label))]


    def random_sample(self, n_shot): ##randomly seperate demonstration set, and evaluate confidence score with leftover instances
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
        self.demonstration_data = {"sentence" : [self.train_sentence[i] for i in data_subsample], 
                           "label" : [self.train_label[i] for i in data_subsample]}
        train_idx = [i for i in range(len(self.train_label))]
        train_excluded = list(set(train_idx) - set(data_subsample))
        self.train_data = {"sentence" : [self.train_sentence[i] for i in train_excluded], "label" : [self.train_label[i] for i in train_excluded], "idx" : train_excluded}

        
    def knn_sample(self, n_shot):
        with open(f'./SBERT_emb/{self.args.data}_train_train_knn_indices.pickle', 'rb') as f:
            knn_indices = pickle.load(f)
        self.demonstration_data = []
        k = n_shot * len(self.id2verb) ## we do not consider stratify class, so add n_shot * n_classes for total demonstrations
        for i in tqdm(range(len(self.train_label))):
            knn_indice = knn_indices[i][:k]
            random.shuffle(knn_indice)
            self.demonstration_data.append({"sentence" : [self.train_sentence[i] for i in knn_indice], "label" : [self.train_label[i] for i in knn_indice]})
        self.train_data['idx'] = [i for i in range(len(self.train_label))]
        

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
        return len(self.train_data['sentence'])
    
    def __getitem__(self, idx):
        prompt = ''
        if self.args.n_shot==0: ## zero_shot, add task instruction
            if self.args.data=='sst2':
                prompt = 'Classify sentiment of a given review sentence in label space [positive, negative]\n'
                prompt+=self.template(self.data['sentence'][idx], mode='inference')
            elif self.args.data=='CR':
                prompt = 'Classify sentiment of a given review sentence in label space [positive, negative]\n'
                prompt+=self.template(self.data['sentence'][idx], mode='inference')
            elif self.args.data=='trec':
                prompt = 'Classify the given question\'s answer type in label space [expression, entity, description, human, location, number]\n'
                prompt+=self.template(self.data['sentence'][idx], mode='inference')
            elif self.args.data=='subj':
                prompt = 'Classify the given input\'s subjectivity in label sapce [objective, subjective]\n'
                prompt+=self.template(self.data['sentence'][idx], mode='inference')
            elif self.args.data=='rte':
                prompt = 'Classify given premise and hypothesis are entailed(true) or not(false) in label space [true, false]'
                prompt+=self.template(self.data['sentence'][idx], mode='inference')
            elif self.args.data=="sst5":
                prompt = 'Classify sentiment of a given review sentence in label space [terrible, bad, neutral, good, great]\n'
                prompt+=self.template(self.data['sentence'][idx], mode='inference')
            elif self.args.data=="agnews":
                prompt = 'Classify topic of a given input in label space [world, sports, business, technology]\n'
                prompt+=self.template(self.data['sentence'][idx], mode='inference')
            label = self.train_data['label'][idx]
            index = self.train_data['idx'][idx]
            return (prompt, label, index)
        if self.args.random:
            for s, l in zip(self.demonstration_data['sentence'], self.demonstration_data['label']):
                prompt+=self.template(s,l, mode='train')
                prompt += '\n'
            prompt+=self.template(self.train_data['sentence'][idx], mode = 'inference')
            label = self.train_data['label'][idx]
            index = self.train_data['idx'][idx]
            return (prompt, label, index)
        else:
            for s, l in zip(self.demonstration_data[idx]['sentence'], self.demonstration_data[idx]['label']):
                prompt+=self.template(s,l, mode='train')
                prompt += '\n'
            prompt+=self.template(self.train_data['sentence'][idx], mode = 'inference')
            label = self.train_data['label'][idx]
            index = self.train_data['idx'][idx]
            return (prompt, label, index)


def main():
    args = parse_args()
    set_seed(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("calculating confidence score from LLM inference model")
    tokenizer = LlamaTokenizer.from_pretrained(args.LLM_model_name_or_path)
    model_config = AutoConfig.from_pretrained(args.LLM_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.LLM_model_name_or_path, config = model_config).to(device)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    dataset = LLM_Confidence_dataset(args)
    id2verb = dataset.id2verb
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle=False)
    result = {}
    for batch in tqdm(dataloader):
        prompt = batch[0]
        label = batch[1].tolist()
        idx = batch[2].tolist()
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device=model.device)
        with torch.no_grad():
            logits = model.forward(input_ids=inputs['input_ids'],
                                    attention_mask=inputs['attention_mask'],
                                    return_dict=True).logits.detach().cpu()
        gen_logits = logits[:, -1, :]
        gen_logits = torch.softmax(gen_logits, dim=-1)
        logits_per_cls = []
        for label_verb in id2verb:
            label_verb_token_id = tokenizer.encode(label_verb)[1] # note the space before label word
            logits_per_cls.append(gen_logits[:, label_verb_token_id])
        prob = torch.stack(logits_per_cls, dim=1).tolist()
        confidence_score = [prob[i][label[i]] for i in range(len(prob))]
        for j in range(len(idx)):
            result[idx[j]] = {"prob" : prob[j], "confidence_score" : confidence_score[j], "label" : label[j]}
    if args.n_shot==0:
        option = f"{args.data}_zeroshot_{args.seed}seed"
    elif args.random:
        option = f"{args.data}_random{args.n_shot}shot_{args.seed}seed"
    else:
        option = f"{args.data}_knn{args.n_shot}shot_{args.seed}seed"
    os.makedirs(f'./LLM_confidence/{args.save_path}', exist_ok=True)
    with open(f'./LLM_confidence/{args.save_path}/'+option+'.json', 'w') as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()