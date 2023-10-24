import argparse
from utils import set_seed
from data import ICL_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="In-Context Learning baseline.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="sst2")
    parser.add_argument("--n_shot", type=int, default=4)
    args = parser.parse_args()

    return args

args = parse_args()
set_seed(args)

ds = ICL_dataset(args)

print(ds[0])