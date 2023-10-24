import torch
from transformers import Trainer, AutoModelForSequenceClassification, AutoTokenizer, LlamaTokenizer, AutoModelForCausalLM, AutoConfig
import argparse
from datasets import load_dataset
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import json
from utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="In-Context Learning confidence score experiment")
    parser.add_argument("--LLM_model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--finetune_model_name_or_path", type=str, default="distilbert-base-uncased-finetuned-sst-2-english")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="sst2")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--finetune", action="store_true", default=False)
    args = parser.parse_args()

    return args

class ConfidenceDataset(Dataset):
    def __init__(self, args):
        self.args = args
        if self.args.data=="sst2":
            self.data = load_dataset(args.data)['train']
            self.id2label = {0: 'negative', 1: 'positive'}
            self.label2id = {'negative': 0, 'positive': 1}
            self.id2verb = ['negative', 'positive']
        elif self.args.data=="CR":
            with open("../data/CR_Train_fixed.json", 'r') as f:
                self.data = json.load(f)
                self.id2label = {0: 'negative', 1: 'positive'}
                self.label2id = {'negative': 0, 'positive': 1}
                self.id2verb = ['negative', 'positive']
        elif self.args.data=="trec":
            self.data = load_dataset(args.data)['train']
            sentence = self.data['text']
            label = self.data['coarse_label']
            self.data = {'sentence' : sentence, 'label' : label}
            self.id2label = {0:'expression', 1:'entity', 2:'description', 3:'human', 4:'location', 5:'number'}
            self.label2id = {v:k for k,v in self.id2label.items()}
            self.id2verb = list(self.label2id.keys())
    def __len__(self):
        return len(self.data['sentence'])
    def template_sst2(self, sentence):
        return f"Review: {sentence}\nSentiment:"
    def template_CR(self, sentence):
        return f"Review: {sentence}\nSentiment:"
    def template_trec(self, sentence):
        return f"Question: {sentence}\nType:"
    def __getitem__(self, idx):
        if self.args.finetune:
            return(self.data['sentence'][idx], self.data['label'][idx])
        else:
            if self.args.data=='sst2':
                prompt = 'Classify sentiment of a given review sentence in label space [positive, negative]\n'
                prompt+=self.template_sst2(self.data['sentence'][idx])
            elif self.args.data=='CR':
                prompt = 'Classify sentiment of a given review sentence in label space [positive, negative]\n'
                prompt+=self.template_CR(self.data['sentence'][idx])
            elif self.args.data=='trec':
                prompt = 'Classify which type of thing the given question is asking about in label space [expression, entity, description, human, location, number]\n'
                prompt+=self.template_trec(self.data['sentence'][idx])
            label = self.data['label'][idx]
            return (prompt, label)
    


def main():
    args = parse_args()
    set_seed(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.finetune:
        print("calculating confidence score from fine-tuned model")
        tokenizer = AutoTokenizer.from_pretrained(args.finetune_model_name_or_path)
        model_config = AutoConfig.from_pretrained(args.finetune_model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(args.finetune_model_name_or_path, config = model_config).to(device)
        model.eval()
        dataset = ConfidenceDataset(args)
        print(len(dataset))
        dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle=False)
        all_prob=[]
        for batch in tqdm(dataloader):
            sentence = batch[0]
            label = batch[1]
            inputs = tokenizer(sentence, return_tensors='pt', padding=True).to(device=model.device)
            with torch.no_grad():
                logits = model(**inputs).logits.detach().cpu()
            prob = torch.softmax(logits, dim=-1).tolist()
            prob=[prob[i][label[i]] for i in range(len(prob))]
            all_prob.extend(prob)
    else:
        print("calculating confidence score from LLM inference model")
        tokenizer = LlamaTokenizer.from_pretrained(args.LLM_model_name_or_path)
        model_config = AutoConfig.from_pretrained(args.LLM_model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(args.LLM_model_name_or_path, config = model_config).to(device)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.eval()
        dataset = ConfidenceDataset(args)
        id2verb = dataset.id2verb
        dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle=False)
        all_prob = []
        for batch in tqdm(dataloader):
            prompt = batch[0]
            label = batch[1]
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device=model.device)
            with torch.no_grad():
                logits = model.forward(input_ids=inputs['input_ids'],
                                        attention_mask=inputs['attention_mask'],
                                        return_dict=True).logits.detach().cpu()
            gen_logits = logits[:, -1, :]
            logits_per_cls = []
            for label_verb in id2verb:
                label_verb_token_id = tokenizer.encode(' ' + label_verb)[-1] # note the space before label word
                logits_per_cls.append(gen_logits[:, label_verb_token_id])
            prob = torch.softmax(torch.stack(logits_per_cls, dim=1), dim=-1).tolist()
            prob = [prob[i][label[i]] for i in range(len(prob))]
            all_prob.extend(prob)
    all_prob = [(i, all_prob[i]) for i in range(len(all_prob))]
    all_prob.sort(key=lambda x : -x[1])
    x = "finetune" if args.finetune else "LLM"
    save_path = f"./prob/{args.data}_{x}_prob.pickle"
    with open(save_path, 'wb') as f:
        pickle.dump(all_prob, f)

if __name__ == "__main__":
    main()


