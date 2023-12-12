from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoConfig, AutoTokenizer
import argparse
from tqdm import tqdm
from utils import set_seed
import torch
from data import ICL_dataset
import os
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="In-Context Learning baseline.")
    parser.add_argument("--llm", type=str, default='llama2_7b')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="sst2")
    parser.add_argument("--n_shot", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--selection", type = str, default="random") #finetune_confidence, LLM_confidence, knn
    parser.add_argument("--LLM_confidence_path", type=str, default='./LLM_confidence')
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
    dataset = ICL_dataset(args)
    id2verb = dataset.id2verb
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle=False)
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
        pred = torch.argmax(torch.stack(prob_per_cls, dim=1), dim=-1).tolist()
        all_pred.extend(pred)
        all_ref.extend(label)
    cnt=0
    for i in range(len(all_pred)):
        if all_pred[i]==all_ref[i]:
            cnt+=1
        else:
            continue
    accuracy = cnt/len(all_pred)
    print(accuracy)
    result_dir = f"./result/{args.llm}"
    os.makedirs(result_dir, exist_ok=True)
    result_file_path = f"{result_dir}/{args.data}_{args.selection}_{args.n_shot}shot.txt"
    if not os.path.isfile(result_file_path):
        with open(result_file_path, 'w') as f:
            f.write(f"{args.seed} : {accuracy}\n")
    else:
        with open(result_file_path, 'a') as f:
            f.write(f"{args.seed} : {accuracy}\n")

if __name__ == "__main__":
    main()




