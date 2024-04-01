import random
from sentence_transformers import SentenceTransformer,util
import torch
import numpy as np
from sklearn.cluster import KMeans

## random sampling without considering 
def random_sampling(datasets, n_shot):
    train_data_by_cls = datasets.train_data_by_cls
    total_train_data_idx = []
    for v in train_data_by_cls.values():
        total_train_data_idx.extend(v)
    num_subsampled_data = n_shot * len(datasets.id2verb)
    data_subsample = random.sample(total_train_data_idx, num_subsampled_data)

    return data_subsample

def random_stratify_sampling(datasets, n_shot):
    train_data_by_cls = datasets.train_data_by_cls
    data_subsample =[]
    for cls in train_data_by_cls.keys():
        data_subsample_by_cls = random.sample(train_data_by_cls, n_shot)
        data_subsample.extend(data_subsample_by_cls)
    random.shuffle(data_subsample)

    return data_subsample

def similar(datasets, n_shot):
    num_subsampled_data = n_shot * len(datasets.id2verb)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
    train_data_by_cls = datasets.train_data_by_cls
    total_train_data_idx = []
    for v in train_data_by_cls.values():
        total_train_data_idx.extend(v)
    total_train_data_sentence = [datasets.train_sentence[i] for i in total_train_data_idx]
    total_dev_data_sentence = datasets.dev_sentence
    print("Start generate embedding")
    train_embedding = model.encode(total_train_data_sentence)
    dev_embedding = model.encode(total_dev_data_sentence)
    print("calculating cosine similarity")
    cosine_scores = np.array(util.cos_sim(dev_embedding, train_embedding))
    sim_idx = np.argsort(cosine_scores, axis=1)[:,::-1]
    demon_idx = sim_idx[:,:num_subsampled_data]
    demon_idx = demon_idx[:,::-1].tolist()
    demon_idx = [[total_train_data_idx[j] for j in l] for l in demon_idx]
    del model
    return demon_idx

def diverse(datasets, n_shot):
    num_subsampled_data = n_shot * len(datasets.id2verb)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
    train_data_by_cls = datasets.train_data_by_cls
    total_train_data_idx = []
    for v in train_data_by_cls.values():
        total_train_data_idx.extend(v)
    total_train_data_sentence = [datasets.train_sentence[i] for i in total_train_data_idx]
    print("Start generate embedding")
    train_embedding = model.encode(total_train_data_sentence)
    del model
    clustering_models = KMeans(n_clusters=num_subsampled_data)
    print("fitting cluster")
    clustering_models.fit(train_embedding)
    cluster_assignment = clustering_models.labels_

    clustered_sentences = [[] for i in range(num_subsampled_data)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(total_train_data_idx[sentence_id])
    data_subsample = [random.choice(l) for l in clustered_sentences]
    random.shuffle(data_subsample)

    return data_subsample





    








