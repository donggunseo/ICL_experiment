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

class ICL_MDL_dataset(Dataset):
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
        self.candidates = 100
        self.num_org = 10
        self.knn_mdl_sample(args.n_shot)
        self.dev_data = {'sentence': [item for item in self.dev_data['sentence'] for _ in range(self.num_org)], 
                         'label' : [item for item in self.dev_data['label'] for _ in range(self.num_org)]}
        
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
    
    def knn_mdl_sample(self, n_shot):
        with open(f'./SBERT_emb/{self.args.data}_train_dev_knn_indices.pickle', 'rb') as f:
            knn_indices = pickle.load(f)
        self.demonstration_data = []
        k = n_shot ## we do not consider stratify class, so add n_shot * n_classes for total demonstrations
        for i in tqdm(range(len(self.dev_data['label']))):
            knn_indice = knn_indices[i][:self.candidates]
            cand = []
            for _ in range(self.num_org):
                data_subsample = random.sample(list(knn_indice), k)
                random.shuffle(data_subsample)
                cand.append({"sentence" : [self.train_sentence[i] for i in data_subsample], "label" : [self.train_label[i] for i in data_subsample]})
            self.demonstration_data.extend(cand)

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
        for s, l in zip(self.demonstration_data[idx]['sentence'], self.demonstration_data[idx]['label']):
            prompt+=self.template(s,l, mode='train')
            prompt += '\n'
        prompt+=self.template(self.dev_data['sentence'][idx], mode = 'inference')
        label = self.dev_data['label'][idx]
        return (prompt, label)

def parse_args():
    parser = argparse.ArgumentParser(description="In-Context Learning baseline.")
    parser.add_argument("--llm", type=str, default='llama2_7b')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="sst2")
    parser.add_argument("--n_shot", type=int, default=20)
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
    dataset = ICL_MDL_dataset(args)
    num_org = dataset.num_org
    id2verb = dataset.id2verb
    dataloader = DataLoader(dataset, batch_size = num_org, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = LlamaTokenizer.from_pretrained(model_dict[args.llm]) if "llama" in args.llm else AutoTokenizer.from_pretrained(model_dict[args.llm])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model_config = AutoConfig.from_pretrained(model_dict[args.llm])
    model = AutoModelForCausalLM.from_pretrained(model_dict[args.llm], config = model_config)
    model.to(device)
    model.eval()
    all_pred = []
    all_ref = []
    for batch in tqdm(dataloader):
        prompt = batch[0]
        label = batch[1].tolist()
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device=model.device)
        with torch.no_grad():
            logits = model.forward(input_ids=inputs['input_ids'],
                                    attention_mask=inputs['attention_mask'],
                                    return_dict=True).logits.detach().cpu()
        gen_logits = logits[:, -1, :]
        prob_per_cls = []
        for label_verb in id2verb:
            if args.llm!="gpt2_xl":
                label_verb_token_id = tokenizer.encode(label_verb)[1] 
            else:
                label_verb_token_id = tokenizer.encode(label_verb)[0]
            prob_per_cls.append(gen_logits[:, label_verb_token_id])
        prob = torch.stack(prob_per_cls, dim=1)
        pred = torch.argmax(prob, dim=-1).tolist()
        prob = torch.softmax(prob, dim=-1).numpy()
        logp = np.log2(prob)
        entropy = np.sum(-prob*logp, axis=-1)
        min_idx = np.argmin(entropy)
        all_pred.append(pred[min_idx])
        all_ref.append(label[0])
    cnt=0
    for i in range(len(all_pred)):
        if all_pred[i]==all_ref[i]:
            cnt+=1
        else:
            continue
    accuracy = cnt/len(all_pred)
    print(accuracy)
    result_dir = f"./result_mdl/{args.llm}"
    os.makedirs(result_dir, exist_ok=True)
    result_file_path = f"{result_dir}/{args.data}_{args.n_shot}shot.txt"
    if not os.path.isfile(result_file_path):
        with open(result_file_path, 'w') as f:
            f.write(f"{args.seed} : {accuracy}\n")
    else:
        with open(result_file_path, 'a') as f:
            f.write(f"{args.seed} : {accuracy}\n")

if __name__ == "__main__":
    main()