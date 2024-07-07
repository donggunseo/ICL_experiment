from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import argparse
from tqdm import tqdm
import torch
import json
import torch.nn as nn
import numpy as np
import random
import os
from sklearn.metrics import accuracy_score, f1_score
from utils import set_seed
from ent_data import entailment_dataset

MODEL_DICT ={
    "llama2_7b" : "meta-llama/Llama-2-7b-hf",
    "llama2_13b" : "meta-llama/Llama-2-13b-hf",
    "opt_6.7b" : "facebook/opt-6.7b",
    "gpt2_xl" : "gpt2-xl"
}

def parse_args():
    parser = argparse.ArgumentParser(description="Calculating hypothesis perplexity")
    parser.add_argument("--llm", type=str, default='llama2_7b')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="rte")
    parser.add_argument("--max_train_size", type=int, default=-1)
    parser.add_argument("--max_dev_size", type=int, default=-1)
    parser.add_argument("--task_instruction", action="store_true")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    set_seed(args)
    datasets = entailment_dataset(args)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT[args.llm])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = AutoConfig.from_pretrained(MODEL_DICT[args.llm])
    model = AutoModelForCausalLM.from_pretrained(MODEL_DICT[args.llm], config= model_config)
    model.to(device)
    model.eval()
    