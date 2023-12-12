import torch
from transformers import  AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import argparse
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import json
from utils import set_seed
import os

def parse_args():
    parser = argparse.ArgumentParser(description="In-Context Learning confidence score experiment")
    parser.add_argument("--finetune_model_name_or_path", type=str, default='./finetune_model/sst2')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="sst2")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    return args

class ConfidenceDataset(Dataset):
    def __init__(self, args):
        self.args = args
        if self.args.data=="sst2":
            self.data = load_dataset(args.data)['train']
            sentence = self.data['sentence']
            label = self.data['label']
            self.data = {'sentence' : sentence, 'label' : label}
            self.id2label = {0: 'negative', 1: 'positive'}
            self.label2id = {v:k for k,v in self.id2label.items()}
            self.id2verb = list(self.label2id.keys())
        elif self.args.data=="CR":
            self.data = load_dataset("SetFit/CR")['train']
            sentence = self.data['text']
            label = self.data['label']
            self.data = {'sentence' : sentence, 'label' : label}
            self.id2label = {0: 'negative', 1: 'positive'}
            self.label2id = {v:k for k,v in self.id2label.items()}
            self.id2verb = list(self.label2id.keys())
        elif self.args.data=="trec":
            self.data = load_dataset(args.data)['train']
            sentence = self.data['text']
            label = self.data['coarse_label']
            self.data = {'sentence' : sentence, 'label' : label}
            self.id2label = {0:'expression', 1:'entity', 2:'description', 3:'human', 4:'location', 5:'number'}
            self.label2id = {v:k for k,v in self.id2label.items()}
            self.id2verb = list(self.label2id.keys())
        elif self.args.data=="subj":
            self.data = load_dataset("SetFit/subj")['train']
            sentence = self.data['text']
            label = self.data['label']
            self.data = {'sentence' : sentence, 'label' : label}
            self.id2label = {0:'objective', 1:'subjective'}
            self.label2id = {v:k for k,v in self.id2label.items()}
            self.id2verb = list(self.label2id.keys())
        elif self.args.data=="sst5":
            self.data = load_dataset("SetFit/sst5")['train']
            sentence = self.data['text']
            label = self.data['label']
            self.data = {'sentence' : sentence, 'label' : label}
            self.id2label = {0: 'terrible', 1: 'bad', 2:'okay', 3:'good', 4:'great'}
            self.label2id = {v:k for k,v in self.id2label.items()}
            self.id2verb = list(self.label2id.keys())
        elif self.args.data=="agnews":
            self.data = load_dataset("ag_news")['train']
            sentence = self.data['text']
            label = self.data['label']
            self.data = {'sentence' : sentence, 'label' : label}
            self.id2label = {0: 'world', 1: 'sports', 2:'business', 3:'technology'}
            self.label2id = {v:k for k,v in self.id2label.items()}
            self.id2verb = list(self.label2id.keys())
    def __len__(self):
        return len(self.data['sentence'])
    def __getitem__(self, idx):
        return(self.data['sentence'][idx], self.data['label'][idx], idx)
    


def main():
    args = parse_args()
    set_seed(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("calculating confidence score from fine-tuned model")
    dataset = ConfidenceDataset(args)
    tokenizer = AutoTokenizer.from_pretrained(args.finetune_model_name_or_path)
    model_config = AutoConfig.from_pretrained(args.finetune_model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.finetune_model_name_or_path, config = model_config).to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle=False)
    result={}
    for batch in tqdm(dataloader):
        sentence = batch[0]
        label = batch[1].tolist()
        idx = batch[2].tolist()
        inputs = tokenizer(sentence, return_tensors='pt', padding=True).to(device=model.device)
        with torch.no_grad():
            logits = model(**inputs).logits.detach().cpu()
        prob = torch.softmax(logits, dim=-1).tolist()
        confidence_score=[prob[i][label[i]] for i in range(len(prob))]
        for j in range(len(idx)):
            result[idx[j]] = {"prob" : prob[j], "confidence_score" : confidence_score[j], "label" : label[j]}
    os.makedirs(f'./Finetune_confidence', exist_ok=True)
    save_path = f'./Finetune_confidence/' + args.data + '_finetune'+ '.json'
    with open(save_path, 'w') as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()


