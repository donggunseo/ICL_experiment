from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
import os
import argparse
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from utils import set_seed, collate_fn
from tqdm import tqdm
import torch.nn as nn

#### Fine-tune model by myself for evaluate Confidence score of training set

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning for obtaining In-Context Learning confidence score experiment")
    parser.add_argument("--finetune_model_name_or_path", type=str, default="distilbert-base-uncased")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="sst2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--save_path", type=str, default = "./finetune_model/")

    args = parser.parse_args()

    return args

class TCDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.finetune_model_name_or_path)
        if self.args.data=="sst2":
            data = load_dataset(args.data)
            if mode == "train":
                self.data = data['train']
            else:
                self.data = data['validation']
            self.id2label = {0: 'negative', 1: 'positive'}
            self.label2id = {v:k for k,v in self.id2label.items()}
            self.id2verb = list(self.label2id.keys())
        elif self.args.data=="CR":
            data = load_dataset("SetFit/CR")
            if mode == "train":
                self.data = data['train']
            else:
                self.data = data['test']
            self.data = {'sentence' : self.data['text'], 'label' : self.data['label']}
            self.id2label = {0: 'negative', 1: 'positive'}
            self.label2id = {v:k for k,v in self.id2label.items()}
            self.id2verb = list(self.label2id.keys())
        elif self.args.data=="trec":
            data = load_dataset(args.data)
            if mode == "train":
                self.data = data['train']
            else:
                self.data = data['test']
            self.data = {'sentence' : self.data['text'], 'label' : self.data['coarse_label']}
            self.id2label = {0:'expression', 1:'entity', 2:'description', 3:'human', 4:'location', 5:'number'}
            self.label2id = {v:k for k,v in self.id2label.items()}
            self.id2verb = list(self.label2id.keys())
        elif self.args.data=="subj":
            data = load_dataset("SetFit/subj")
            if mode == "train":
                self.data = data['train']
            else:
                self.data = data['test']
            self.data = {'sentence' : self.data['text'], 'label' : self.data['label']}
            self.id2label = {0:'objective', 1:'subjective'}
            self.label2id = {v:k for k,v in self.id2label.items()}
            self.id2verb = list(self.label2id.keys())
        elif self.args.data=="sst5":
            data = load_dataset("SetFit/sst5")
            if mode == "train":
                self.data = data['train']
            else:
                self.data = data['test']
            self.data = {'sentence' : self.data['text'], 'label' : self.data['label']}
            self.id2label = {0: 'terrible', 1: 'bad', 2:'neutral', 3:'good', 4:'great'}
            self.label2id = {v:k for k,v in self.id2label.items()}
            self.id2verb = list(self.label2id.keys())
        elif self.args.data=="agnews":
            data = load_dataset("ag_news")
            if mode == "train":
                self.data = data['train']
            else:
                self.data = data['test']
            self.data = {'sentence' : self.data['text'], 'label' : self.data['label']}
            self.id2label = {0: 'world', 1: 'sports', 2:'business', 3:'technology'}
            self.label2id = {v:k for k,v in self.id2label.items()}
            self.id2verb = list(self.label2id.keys())

    def __len__(self):
        return len(self.data['sentence'])
    
    def __getitem__(self, idx):
        s = self.data['sentence'][idx]
        s = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(s))
        s = [101]+s+[102]
        return (s, self.data['label'][idx])


def main():
    args = parse_args()
    args.save_path+=args.data
    set_seed(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = TCDataset(args)
    val_dataset = TCDataset(args, mode = "eval")
    num_classes = len(train_dataset.id2verb)
    tokenizer = AutoTokenizer.from_pretrained(args.finetune_model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.finetune_model_name_or_path, num_labels = num_classes).to(device)

    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, collate_fn=collate_fn)
    total_steps = int(len(train_dataloader) * args.epoch)
    warmup_steps = int(total_steps * args.warmup_ratio)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    os.makedirs(args.save_path, exist_ok=True)

    best_acc = -1
    for epoch in range(args.epoch):
        print(f"Epoch {epoch}")
        model.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            input_ids = batch[0].to(device)
            input_mask = batch[1].to(device)
            inputs = {'input_ids' : input_ids, 'attention_mask' : input_mask}
            label = batch[2].to(device)
            outputs = model(**inputs).logits
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            if step%10==0:
                print(f"loss : {loss.item()}, step : {step}")
        all_crt = []
        for batch_v in tqdm(val_dataloader):
            model.eval()
            input_ids = batch_v[0].to(device)
            input_mask = batch_v[1].to(device)
            inputs = {'input_ids' : input_ids, 'attention_mask' : input_mask}
            label = batch_v[2]
            with torch.no_grad():
                logits = model(**inputs).logits.detach().cpu()
            pred = torch.argmax(logits, dim=-1).tolist()
            crt = [1 if l==p else 0 for l,p in zip(label, pred)]
            all_crt.extend(crt)
        acc = sum(all_crt)/len(all_crt) * 100

        if not os.path.isfile(f"{args.save_path}/training_result.txt"):
            with open(f"{args.save_path}/training_result.txt", 'w') as f:
                f.write(f"epoch {epoch} : {acc}\n")
        else:
            with open(f"{args.save_path}/training_result.txt", 'a') as f:
                f.write(f"epoch {epoch} : {acc}\n")
        
        if acc>best_acc:
            best_acc = acc
            model.save_pretrained(f"{args.save_path}")
            tokenizer.save_pretrained(f"{args.save_path}")

if __name__ == "__main__":
    main()






            

