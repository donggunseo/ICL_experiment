import random
import numpy as np
import torch

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def collate_fn(batch):
    max_len = max([len(f[0]) for f in batch])
    input_ids = [f[0] + [0] * (max_len - len(f[0])) for f in batch]
    input_mask = [[1] * len(f[0]) + [0] * (max_len - len(f[0])) for f in batch]
    labels = [f[1] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    output = (input_ids, input_mask, labels)
    return output