from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import argparse
from tqdm import tqdm
from utils import set_seed
import torch
from data import ICL_dataset
import json
from torch.utils.data import DataLoader
from demonstration_selection import random_sampling, random_stratify_sampling, similar, diverse
import torch.nn as nn
import numpy as np
import random

model_dict ={
    "llama2_7b" : "meta-llama/Llama-2-7b-hf",
    "llama2_13b" : "meta-llama/Llama-2-13b-hf",
    "opt_6.7b" : "facebook/opt-6.7b",
    "gpt2_xl" : "gpt2-xl"
}

SOFTMAX = nn.Softmax(dim=-1)

def icl_inference(model, prompt, id2verb, tokenizer):
    inputs = tokenizer(prompt, return_tensors = 'pt').to(device=model.device)
    with torch.no_grad():
        logits = model.forward(input_ids=inputs['input_ids'], 
                               attention_mask=inputs['attention_mask'],
                               return_dict=True).logits.detach().cpu()
    gen_logits = logits[:,-1,:]
    gen_logits = SOFTMAX(gen_logits)
    prob_per_cls = []
    for label_verb in id2verb:
        label_verb_token_id = tokenizer.encode(label_verb, add_special_tokens=False)[0]
        prob_per_cls.append(gen_logits[:, label_verb_token_id].item())
    return prob_per_cls

def parse_args():
    parser = argparse.ArgumentParser(description="In-Context Learning baseline.")
    parser.add_argument("--llm", type=str, default='llama2_7b')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="sst2")
    parser.add_argument("--n_shot", type=int, default=4)
    parser.add_argument("--max_train_size", type=int, default=10000)
    parser.add_argument("--max_dev_size", type=int, default=100)
    parser.add_argument("--selection", type=str, default = 'random_sampling')
    args = parser.parse_args()
    return args

def prompt_generation(datasets, demon_idx):
    template = datasets.template
    train_sentence = datasets.train_sentence
    train_label = datasets.train_label
    if type(demon_idx[0])==list: ##similar
        prompt_list = []
        for d in demon_idx:
            prompt = ''
            for t_idx in d:
                ts = train_sentence[t_idx]
                tl = train_label[t_idx]
                prompt+=template(ts,tl,mode='train')
                prompt += '\n'
            prompt_list.append(prompt)
    else:
        prompt = ''
        for t_idx in demon_idx:
            ts = train_sentence[t_idx]
            tl = train_label[t_idx]
            prompt+=template(ts,tl,mode='train')
            prompt += '\n'
        prompt_list = [prompt]*len(datasets)
    return prompt_list
        


def main():
    args = parse_args()
    set_seed(args)
    datasets = ICL_dataset(args)
    dataloader = DataLoader(datasets, batch_size = 1, shuffle=False)
    id2verb = datasets.id2verb


    if args.selection == 'random_sampling':
        demon_idx = random_sampling(datasets, args.n_shot)
    elif args.selection == 'random_stratify_sampling':
        demon_idx = random_stratify_sampling(datasets, args.n_shot)
    elif args.selection == 'similar':
        demon_idx = similar(datasets, args.n_shot)
    elif args.selection == 'diverse':
        demon_idx = diverse(datasets, args.n_shot)

    prompt_list = prompt_generation(datasets, demon_idx)
    tokenizer = AutoTokenizer.from_pretrained(model_dict[args.llm])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = AutoConfig.from_pretrained(model_dict[args.llm])
    model = AutoModelForCausalLM.from_pretrained(model_dict[args.llm], config = model_config)
    model.to(device)
    model.eval()

    cnt = 0
    for i, d in tqdm(enumerate(dataloader)):
        prompt = prompt_list[i] + d[0][0]
        label = d[1][0]
        prob_per_cls = icl_inference(model, prompt, id2verb, tokenizer)
        prediction = np.argmax(np.array(prob_per_cls))
        if label==prediction:
            cnt+=1
    acc = cnt / len(datasets)
    print(f"Accuracy : {acc}")


def main2():
    args = parse_args()
    set_seed(args)
    datasets = ICL_dataset(args)
    dataloader = DataLoader(datasets, batch_size = 1, shuffle=False)
    id2verb = datasets.id2verb
    
    #####
    train_data_by_cls = datasets.train_data_by_cls
    total_train_data_idx = []
    for v in train_data_by_cls.values():
        total_train_data_idx.extend(v)
    D_selected = random.sample(total_train_data_idx, 100)
    leftover = list(set(total_train_data_idx).difference(set(D_selected)))
    sample_prompt_idx = random.sample(leftover, 7)

    template = datasets.template
    train_sentence = datasets.train_sentence
    train_label = datasets.train_label
    
    prompt_one_shot_list = []
    for d_idx in D_selected:
        prompt = template(train_sentence[d_idx], train_label[d_idx], mode='train')
        prompt += '\n'
        prompt_one_shot_list.append(prompt)
    
    prompt_k_1_shot = ''
    for d_idx in sample_prompt_idx:
        prompt_k_1_shot+=template(train_sentence[d_idx], train_label[d_idx], mode='train')
        prompt_k_1_shot += '\n'

    prompt_k_shot_list = []
    for one_p in prompt_one_shot_list:
        prompt_k_shot_list.append(prompt_k_1_shot+one_p+'\n')
    
    tokenizer = AutoTokenizer.from_pretrained(model_dict[args.llm])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = AutoConfig.from_pretrained(model_dict[args.llm])
    model = AutoModelForCausalLM.from_pretrained(model_dict[args.llm], config = model_config)
    model.to(device)
    model.eval()

    ## zero_shot
    pred_prob_zero = []
    for i, d in tqdm(enumerate(dataloader)):
        prompt = d[0][0]
        label = d[1][0]
        prob_per_cls = icl_inference(model, prompt, id2verb, tokenizer)
        pred_prob_zero.append(prob_per_cls[label])
    ## one_shot
    pred_prob_one = []
    for p in tqdm(prompt_one_shot_list):
        prob_per_prompt = []
        for i, d in tqdm(enumerate(dataloader)):
            prompt = p+d[0][0]
            label = d[1][0]
            prob_per_cls = icl_inference(model, prompt, id2verb, tokenizer)
            prob_per_prompt.append(prob_per_cls[label])
        pred_prob_one.append(prob_per_prompt)
    
    one_influence = {}
    for i, item in enumerate(pred_prob_one):
        diff = [o-z for o,z in zip(item, pred_prob_zero)]
        one_influence[D_selected[i]] = sum(diff)/len(diff)
    ##save 1-shot - 0-shot influence
    with open(f'./{args.data}_one_influence.json', 'w') as f:
        json.dump(one_influence,f)
    
    ## k-1 shot
    pred_prob_k_1 = []
    for i, d in tqdm(enumerate(dataloader)):
        prompt = prompt_k_1_shot + d[0][0]
        label = d[1][0]
        prob_per_cls = icl_inference(model, prompt, id2verb, tokenizer)
        pred_prob_k_1.append(prob_per_cls[label])
    ## k-shot
    pred_prob_k=[]
    for p in tqdm(prompt_k_shot_list):
        prob_per_prompt = []
        for i, d in tqdm(enumerate(dataloader)):
            prompt = p+d[0][0]
            label = d[1][0]
            prob_per_cls = icl_inference(model, prompt, id2verb, tokenizer)
            prob_per_prompt.append(prob_per_cls[label])
        pred_prob_k.append(prob_per_prompt)
    
    k_influence = {}
    for i, item in enumerate(pred_prob_k):
        diff = [o-z for o,z in zip(item, pred_prob_k_1)]
        k_influence[D_selected[i]] = sum(diff)/len(diff)
    with open(f'./{args.data}_k_influence.json', 'w') as f:
        json.dump(k_influence,f)


if __name__ == "__main__":
    main2()
