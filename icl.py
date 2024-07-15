from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
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

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = []):
        StoppingCriteria.__init__(self)
    def __call__(self, input_ids:torch.LongTensor, scores:torch.FloatTensor, stops=[]):
        self.stops = stops
        for i in range(len(stops)):
            self.stops = self.stops[i]

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
    top_k_results = {tokenizer.decode(i.item(), skip_special_token=True):round(gen_logits[0][i.item()].item(),4) for i in top_k_indices[:10]}
    prob_per_cls = []
    for label_verb in id2verb:
        label_verb_token_id = tokenizer.encode(":"+label_verb, add_special_tokens=False)[1]
        prob_per_cls.append(gen_logits[0][label_verb_token_id].item())
    return prob_per_cls, top_k_results

def icl_generate(model, prompt, tokenizer, stopping_criteria):
    inputs = tokenizer(prompt, return_tensors = 'pt').to(device=model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id, eos_token_id=13)
    decoded = tokenizer.batch_decode(gen[:,inputs['input_ids'].shape[1]:-1], add_special_tokens = False)[0]
    print(decoded)
    return decoded

def parse_args():
    parser = argparse.ArgumentParser(description="In-Context Learning baseline.")
    parser.add_argument("--llm", type=str, default='llama2_7b')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="sst2")
    parser.add_argument("--n_shot", type=int, default=4)
    parser.add_argument("--max_train_size", type=int, default=-1)
    parser.add_argument("--max_dev_size", type=int, default=-1)
    parser.add_argument("--selection", type=str, default = 'random_stratify_sampling')
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

def similarity_distribution():
    args = parse_args()
    datasets = ICL_dataset(args)
    dataloader = DataLoader(datasets, batch_size = 1, shuffle=False)
    id2verb = datasets.id2verb

    cosine_scores= demon_select(datasets, args)
    with open(f"./{args.data}_cosine_scores.json", 'w') as f:
        json.dump(cosine_scores,f)

def icl_opengen():
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
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops = [[13]])])


    result_file_name = f"./result_json/{args.data}_gen_results.json"
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
        "train_dev_size" : (len(datasets.train_data.keys()), len(datasets.dev_data.keys())),
        "results": {}
        }

    correct = 0
    for i, d in tqdm(enumerate(dataloader)):
        test_idx = d[0][0].item()
        demon_idx = prompt_list[i][1]
        context = prompt_list[i][0]
        prompt = context + d[1][0]
        label = d[2][0].item()
        decoded = icl_generate(model, prompt, tokenizer, stopping_criteria)
        result_dict['results'][test_idx] = {
            "test_input" : d[1][0], 
            "demonstration_idx" : demon_idx, 
            "demonstration_label" : [id2verb[datasets.train_data[d][1]] for d in demon_idx],
            "answer" : id2verb[label], 
            "prediction" :decoded,
            }
        if "similar" in args.selection:
            result_dict['results'][test_idx]['rank'] = rank[i]
        if id2verb[label]==decoded:
            correct+=1
    acc = correct/result_dict['train_dev_size'][1]
    result_dict['accuracy'] = round(acc,5)

    result_history.append(result_dict)

    with open(result_file_name, 'w') as f:
        json.dump(result_history, f)

def main():
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

    result_file_name = f"./result_entail_json/{args.data}_results.json"
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
            "test_input" : d[1][0], 
            "demonstration_idx" : demon_idx, 
            "demonstration_label" : [id2verb[datasets.train_data[d][1]] for d in demon_idx],
            "answer" : id2verb[label], 
            "prediction" :id2verb[prediction],
            "prob_distribution" : {id2verb[i]:round(p,4) for i,p in enumerate(prob_per_cls)},
            "top10_token" : top_k_results
            }
        if "similar" in args.selection:
            result_dict['results'][test_idx]['rank'] = rank[i]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average = "macro")
    result_dict['accuracy'] = round(acc,5)
    result_dict['macro_f1_score'] = round(f1,5)

    result_history.append(result_dict)

    with open(result_file_name, 'w') as f:
        json.dump(result_history, f)



if __name__ == "__main__":
    main()
