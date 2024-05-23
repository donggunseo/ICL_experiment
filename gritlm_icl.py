from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import argparse
from tqdm import tqdm
from utils import set_seed
import torch
from data import ICL_dataset
import json
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import random
from gritlm import GritLM
from numpy import dot
from numpy.linalg import norm
import pickle
from sentence_transformers import SentenceTransformer,util


def parse_args():
    parser = argparse.ArgumentParser(description="In-Context Learning for GritLM experiment.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="sst2")
    parser.add_argument("--n_shot", type=int, default=8)
    parser.add_argument("--max_train_size", type=int, default=10000)
    parser.add_argument("--max_dev_size", type=int, default=1000)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args)
    datasets = ICL_dataset(args)
    id2verb = datasets.id2verb

    model = GritLM("GritLM/GritLM-7B", torch_dtype="auto")
    model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    
    queries = [l[0] for l in datasets.dev_data.values()]
    q_idx = list(datasets.dev_data.keys())
    documents = [l[0] for l in datasets.train_data.values()]
    d_idx = list(datasets.train_data.keys())
    d_rep = model.encode(documents, batch_size=2, instruction="<|embed|>\n")
    q_rep = model.encode(queries, batch_size=2, instruction="<|embed|>\n")

    def cos_sim(A,B):
        return dot(A, B)/(norm(A)*norm(B))
    
    sims = {q: [cos_sim(q_rep[i], d_rep[j]) for j in range(len(d_rep))] for i, q in tqdm(enumerate(queries))}

    template = datasets.template
    prediction_gt = []
    for i, (q, q_sims) in tqdm(enumerate(sims.items())):
        sim_idx = np.argpartition(q_sims, -8)[-8:]
        prompt = ''
        for s in sim_idx:
            prompt+=template(datasets.train_data[d_idx[s]][0], label = datasets.train_data[d_idx[s]][1])
            prompt+='\n'
        prompt+=template(q, mode="inference")
        answer = id2verb[datasets.dev_data[q_idx[i]][1]]
        messages = [{'role':"user", "content":prompt}]
        encoded = model.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        encoded = {'input_ids': encoded.to(model.device)}
        encoded['attention_mask'] = torch.ones((1,encoded['input_ids'].shape[1]), dtype = torch.long, device = model.device)
        gen = model.generate(**encoded, max_new_tokens=256, do_sample=False)
        decoded = model.tokenizer.batch_decode(gen[:,encoded['input_ids'].shape[1]:], add_special_tokens = False)
        decoded = decoded[0].replace('</s>', '')
        prediction_gt.append((answer, decoded))
    cnt = 0
    for item in prediction_gt:
        if item[0]==item[1]:
            cnt+=1
    cnt_f=0
    for item in prediction_gt:
        if item[1] not in id2verb:
            cnt_f+=1
    
    print(f'Accuracy for {args.data} is {cnt/len(prediction_gt)} using GritLM embedding\n')
    print(f'OOS and ISOOF label prediction ratio is {cnt_f/len(prediction_gt)}\n\n\n')

    print('Do KNN retrieval on SBERT')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SBERT_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
    d_rep = SBERT_model.encode(documents)
    q_rep = SBERT_model.encode(queries)
    del SBERT_model
    sims2 = {q: [cos_sim(q_rep[i], d_rep[j]) for j in range(len(d_rep))] for i, q in tqdm(enumerate(queries))}
    prediction_gt2 = []
    for i, (q, q_sims) in tqdm(enumerate(sims2.items())):
        sim_idx = np.argpartition(q_sims, -8)[-8:]
        prompt = ""
        for s in sim_idx:
            prompt+=template(datasets.train_data[d_idx[s]][0], label = datasets.train_data[d_idx[s]][1])
            prompt+='\n'
        prompt+=template(q, mode="inference")
        answer = id2verb[datasets.dev_data[q_idx[i]][1]]
        messages = [{'role':"user", "content":prompt}]
        encoded = model.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        encoded = {'input_ids': encoded.to(model.device)}
        encoded['attention_mask'] = torch.ones((1,encoded['input_ids'].shape[1]), dtype = torch.long, device = model.device)
        gen = model.generate(**encoded, max_new_tokens=256, do_sample=False)
        decoded = model.tokenizer.batch_decode(gen[:,encoded['input_ids'].shape[1]:], add_special_tokens = False)
        decoded = decoded[0].replace('</s>', '')
        prediction_gt2.append((answer, decoded))
    
    cnt2 = 0
    for item in prediction_gt2:
        if item[0]==item[1]:
            cnt2+=1
    cnt_f2=0
    for item in prediction_gt2:
        if item[1] not in id2verb:
            cnt_f2+=1
    
    print(f'Accuracy for {args.data} is {cnt2/len(prediction_gt2)} using SBERT embedding\n')
    print(f'OOS and ISOOF label prediction ratio is {cnt_f2/len(prediction_gt2)}\n\n\n')


if __name__ == "__main__":
    main()





