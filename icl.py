from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoConfig
import argparse
from tqdm import tqdm
from utils import set_seed
import torch
from data import ICL_dataset
import os
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="In-Context Learning baseline.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="sst2")
    parser.add_argument("--n_shot", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--confidence", action="store_true", default=False)
    parser.add_argument("--confidence_finetune", action="store_true", default=False)
    # parser.add_argument("--confidence_prob_path", type=str, default='./sst2_finetune_prob.pickle')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    confi_option = "finetune" if args.confidence_finetune else "LLM"
    args.confidence_prob_path = f"./prob/{args.data}_{confi_option}_prob.pickle"
    set_seed(args)
    dataset = ICL_dataset(args)
    id2verb = dataset.id2verb
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model_config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, config = model_config)
    model.to(device)
    model.eval()
    all_pred = []
    all_ref = []
    for batch in tqdm(dataloader):
        prompt = batch[0]
        label = batch[1]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device=model.device)
        with torch.no_grad():
            logits = model.forward(input_ids=inputs['input_ids'],
                                    attention_mask=inputs['attention_mask'],
                                    return_dict=True).logits.detach().cpu()
        gen_logits = logits[:, -1, :]
        gen_prob = torch.softmax(gen_logits, dim=-1)
        prob_per_cls = []
        for label_verb in id2verb:
            label_verb_token_id = tokenizer.encode(' ' + label_verb)[-1] # note the space before label word
            prob_per_cls.append(gen_prob[:, label_verb_token_id])
        pred = torch.argmax(torch.stack(prob_per_cls, dim=1), dim=-1).tolist()
        all_pred.extend(pred)
        all_ref.extend(label)
        if len(pred)!=len(label):
            print("error")
            break
    cnt=0
    for i in range(len(all_pred)):
        if all_pred[i]==all_ref[i]:
            cnt+=1
        else:
            continue
    accuracy = cnt/len(all_pred)
    print(accuracy)
    option = f"confidence_{confi_option}" if args.confidence else "random"
    result_file_path = f"./result/{args.data}_{option}_{args.n_shot}shot.txt"
    if not os.path.isfile(result_file_path):
        with open(result_file_path, 'w') as f:
            f.write(f"{args.seed} : {accuracy}\n")
    else:
        with open(result_file_path, 'a') as f:
            f.write(f"{args.seed} : {accuracy}\n")

if __name__ == "__main__":
    main()




