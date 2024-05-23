from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import argparse
from tqdm import tqdm
from utils import set_seed
import torch
from data import ICL_dataset
import json
from torch.utils.data import DataLoader
from demonstration_selection import demon_select
import torch.nn as nn
import numpy as np
import random
import os
from sklearn.metrics import accuracy_score, f1_score
from huggingface_hub import login

MODEL_DICT ={
    "llama2_7b" : "meta-llama/Llama-2-7b-hf",
    "llama2_13b" : "meta-llama/Llama-2-13b-hf",
    "opt_6.7b" : "facebook/opt-6.7b",
    "gpt2_xl" : "gpt2-xl"
}

SOFTMAX = nn.Softmax(dim=-1)

def icl_inference(model, prompt, id2verb, tokenizer, args):
    inputs = tokenizer(prompt, return_tensors = 'pt').to(device=model.device)
    with torch.no_grad():
        logits = model.forward(input_ids=inputs['input_ids'], 
                               attention_mask=inputs['attention_mask'],
                               return_dict=True).logits.detach().cpu()
    gen_logits = logits[:,-1,:]
    gen_logits = SOFTMAX(gen_logits)
    _, top_k_indices = torch.sort(gen_logits[0], descending=True)
    top_k_results = {tokenizer.decode(i.item(), skip_special_token=True):round(gen_logits[0][i].item(),4) for i in top_k_indices[:10]}
    prob_per_cls = []
    for label_verb in id2verb:
        label_verb_token_id = tokenizer.encode("..."+label_verb, add_special_tokens=False)[1]
        prob_per_cls.append(gen_logits[0][label_verb_token_id].item())
    return prob_per_cls, top_k_results

def icl_generate(model, prompt, tokenizer):
    inputs = tokenizer(prompt, return_tensors = 'pt').to(device=model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(gen[:,inputs['input_ids'].shape[1]:], add_special_tokens = False)[0]
    return decoded

def parse_args():
    parser = argparse.ArgumentParser(description="In-Context Learning baseline.")
    parser.add_argument("--llm", type=str, default='llama2_7b')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="sst2")
    parser.add_argument("--n_shot", type=int, default=4)
    parser.add_argument("--max_train_size", type=int, default=10000)
    parser.add_argument("--max_dev_size", type=int, default=100)
    parser.add_argument("--selection", type=str, default = 'random_sampling')
    parser.add_argument("--task_instruction", action="store_true")
    parser.add_argument("--n_filler", type=int, default=0)
    args = parser.parse_args()
    return args

def prompt_generation(datasets, demon_idx, args):
    template = datasets.template
    prompt_list = [datasets.inst if args.task_instruction else "" for _ in range(len(datasets)) ]
    temp = np.array(demon_idx)
    if len(temp.shape)==1:
        prompt = ""
        for d in demon_idx:
            t, l = datasets.train_data[d]
            prompt+=template(t, l, args.n_filler)
            prompt+="\n"
        prompt_list = [(inst + prompt, demon_idx) for inst in prompt_list]
    else: ## similar
        for i, d_list in enumerate(demon_idx):
            prompt = ""
            for d in d_list:
                t, l = datasets.train_data[d]
                prompt+=template(t, l, args.n_filler)
                prompt+="\n"
            prompt_list[i] = (prompt_list[i]+prompt, demon_idx[i])
    return prompt_list

def main():
    # login(token='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')
    args = parse_args()
    set_seed(args)
    datasets = ICL_dataset(args)
    dataloader = DataLoader(datasets, batch_size = 1, shuffle=False)
    id2verb = datasets.id2verb

    if "similar" in args.selection:
        demon_idx, rank = demon_select(datasets, args)
    else:
        demon_idx = demon_select(datasets, args)

    prompt_list = prompt_generation(datasets, demon_idx, args)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT[args.llm])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = AutoConfig.from_pretrained(MODEL_DICT[args.llm])
    model = AutoModelForCausalLM.from_pretrained(MODEL_DICT[args.llm], config= model_config)
    model.to(device)
    model.eval()

    result_file_name = f"./{args.data}_results_retrieval.json"
    if os.path.isfile(result_file_name):
        with open(result_file_name, 'r') as f:
            result_history = json.load(f)
    else:
        result_history=[]
    ## result_history is list type
    result_dict = {
        "model": args.llm, 
        "seed" : args.seed, 
        "n_shot" : args.n_shot, 
        "selection" : args.selection, 
        "task_instruction": args.task_instruction, 
        # "number of filler tokens" : args.n_filler, 
        "train_dev_size" : (len(datasets.train_data.keys()), len(datasets.dev_data.keys())),
        "results": {}
        }

    y_true = []
    y_pred = []
    for i, d in tqdm(enumerate(dataloader)):
        test_idx = d[0][0].item()
        demon_idx = prompt_list[i][1]
        context = prompt_list[i][0]
        prompt = context + d[1][0]
        label = d[2][0].item()
        prob_per_cls, top_k_results= icl_inference(model, prompt, id2verb, tokenizer, args)
        prediction = np.argmax(np.array(prob_per_cls))
        y_true.append(label)
        y_pred.append(prediction)
        result_dict['results'][test_idx] = {
            "test_input" : datasets.dev_data[test_idx][0], 
            "demonstration_idx" : demon_idx, 
            "demonstration_label" : [id2verb[datasets.train_data[d][1]] for d in demon_idx],
            "rank" : rank[i],
            "answer" : id2verb[label], 
            "prediction" :id2verb[prediction],
            "prob_distribution" : {id2verb[i]:round(p,4) for i,p in enumerate(prob_per_cls)},
            "top10_token" : top_k_results
            }
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average = "macro")
    result_dict['accuracy'] = round(acc,5)
    result_dict['macro_f1_score'] = round(f1,5)

    result_history.append(result_dict)

    with open(result_file_name, 'w') as f:
        json.dump(result_history, f)

def influence_validation():
    args = parse_args()
    set_seed(args)
    datasets = ICL_dataset(args)
    dataloader = DataLoader(datasets, batch_size = 1, shuffle=False)
    id2verb = datasets.id2verb

    with open(f"./{args.data}_one_influence.json", 'r') as f:
        data_influence_score = json.load(f)
    ## 100개중에 top-4 하나 뽑고 나머지 15개 org 랜덤 (맨 앞이 top-4)
    data_influence_score_sorted = sorted(data_influence_score.items(), key=lambda item: item[1], reverse=True)
    d_idx_list = []
    top_k_idx = (sorted([int(l[0]) for l in data_influence_score_sorted[:4]],reverse=True), [3,2,1,0])
    d_idx_list.append(top_k_idx)
    for _ in range(15):
        t = random.sample(list(enumerate(data_influence_score_sorted)), 4)
        t = (sorted([int(l[1][0]) for l in t],reverse=True), sorted([l[0] for l in t],reverse=True))
        d_idx_list.append(t)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT[args.llm])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = AutoConfig.from_pretrained(MODEL_DICT[args.llm])
    model = AutoModelForCausalLM.from_pretrained(MODEL_DICT[args.llm], config = model_config)
    model.to(device)
    model.eval()
    total_acc = []
    for demon_idx, inf_score_order in tqdm(d_idx_list):
        prompt_list = prompt_generation(datasets, demon_idx)
        cnt = 0
        for i, d in enumerate(dataloader):
            prompt = prompt_list[i] + d[0][0]
            label = d[1][0]
            prob_per_cls = icl_inference(model, prompt, id2verb, tokenizer)
            prediction = np.argmax(np.array(prob_per_cls))
            if label==prediction:
                cnt+=1
        acc = cnt / len(datasets)
        print(f"Influence Score Order : {inf_score_order}, Accuracy : {acc}")
        total_acc.append((inf_score_order, acc))
    with open(f'./{args.data}_inf_acc.json', 'w') as f:
        json.dump(total_acc,f)

def influence_measure():
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
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT[args.llm])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = AutoConfig.from_pretrained(MODEL_DICT[args.llm])
    model = AutoModelForCausalLM.from_pretrained(MODEL_DICT[args.llm], config = model_config)
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
    main()
