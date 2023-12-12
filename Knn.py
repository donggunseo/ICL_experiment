import torch
from sentence_transformers import SentenceTransformer
from utils import set_seed
import argparse
from datasets import load_dataset
import pickle
import faiss
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Obtaining KNN instances experiment")
    parser.add_argument("--embedding_model_path", type=str, default='sentence-transformers/all-mpnet-base-v2')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="sst2")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    set_seed(args)
    
    if args.data=="CR":
        data = load_dataset("SetFit/CR")
        train_data = data['train']
        dev_data = data['test']
        train_data = {'sentence' : train_data['text'], 'label' : train_data['label']}
        dev_data = {'sentence' : dev_data['text'], 'label' : dev_data['label']}
    elif args.data=="sst2":
        data = load_dataset(args.data)
        train_data = data['train']
        dev_data = data['validation']
    elif args.data=="trec":
        data = load_dataset(args.data)
        train_data = data['train']
        dev_data = data['test']
        train_data = {'sentence' : train_data['text'], 'label' : train_data['coarse_label']}
        dev_data = {'sentence' : dev_data['text'], 'label' : dev_data['coarse_label']}
    elif args.data=='subj':
        data = load_dataset("SetFit/subj")
        train_data = data['train']
        dev_data = data['test']
        train_data = {'sentence' : train_data['text'], 'label' : train_data['label']}
        dev_data = {'sentence' : dev_data['text'], 'label' : dev_data['label']}
    elif args.data=='rte':
        data = load_dataset("glue", 'rte')
        train_data = data['train']
        dev_data = data['validation']
        sentence1 = train_data['sentence1']
        sentence2 = train_data['sentence2']
        concat_s = [s1+" "+s2 for s1, s2 in zip(sentence1, sentence2)]
        train_data = {'sentence' : concat_s, 'label' : train_data['label']}
        sentence1 = dev_data['sentence1']
        sentence2 = dev_data['sentence2']
        concat_s = [s1+" "+s2 for s1, s2 in zip(sentence1, sentence2)]
        dev_data = {'sentence' : concat_s, 'label' : dev_data['label']}
    elif args.data=='sst5':
        data = load_dataset("SetFit/sst5")
        train_data = data['train']
        dev_data = data['test']
        train_data = {'sentence' : train_data['text'], 'label' : train_data['label']}
        dev_data = {'sentence' : dev_data['text'], 'label' : dev_data['label']}
    elif args.data=="agnews":
        data = load_dataset("ag_news")
        train_data = data['train']
        dev_data = data['test']
        train_data = {'sentence' : train_data['text'], 'label' : train_data['label']}
        dev_data = {'sentence' : dev_data['text'], 'label' : dev_data['label']}
    
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("start generate embedding")
    train_embedding = model.encode(train_data['sentence'])
    dev_embedding = model.encode(dev_data['sentence'])
    with open(f"./SBERT_emb/{args.data}_train_emb.pickle", 'wb') as f:
        pickle.dump(train_embedding, f)
    with open(f"./SBERT_emb/{args.data}_dev_emb.pickle", 'wb') as f:
        pickle.dump(dev_embedding, f)

    print("insert embedding into faiss")
    dimension = len(train_embedding[0])
    num_training = len(train_embedding)
    num_dev = len(dev_embedding)
    

    train_embedding = np.array(train_embedding).astype('float32')
    dev_embedding = np.array(dev_embedding).astype('float32')
    Index = faiss.IndexFlatIP(dimension)
    Index.add(train_embedding)
    
    print("generate knn train-dev")
    k = num_training if num_training <= 80000 else 10000 ## agnews was set to 10000
    distances, indices = Index.search(dev_embedding, k)
    train_dev_knn = {i : indices[i] for i in range(len(dev_embedding))}
    
    print("generate knn train-train")
    distances, indices = Index.search(train_embedding, k)
    train_train_knn = {i : indices[i][1:] for i in range(len(train_embedding))}
    os.makedirs('./SBERT_emb', exist_ok = True)

    with open(f"./SBERT_emb/{args.data}_train_dev_knn_indices.pickle", 'wb') as f:
        pickle.dump(train_dev_knn, f)
    
    with open(f"./SBERT_emb/{args.data}_train_train_knn_indices.pickle", 'wb') as f:
        pickle.dump(train_train_knn, f)

if __name__ == "__main__":
    main()
    

    


                          

    