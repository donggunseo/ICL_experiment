import random
from sentence_transformers import SentenceTransformer,util
import torch
import numpy as np
from sklearn.cluster import KMeans
import sys


def demon_select(datasets, args):
    if args.selection == "random_sampling":
        data_subsample = random.sample(list(datasets.train_data.keys()), args.n_shot)
    elif args.selection == "random_stratify_sampling":
        data_by_cls = {c:[] for c in datasets.id2label.keys()}
        for idx in datasets.train_data.keys():
            data_by_cls[datasets.train_data[idx][1]].append(idx)
        quotient = args.n_shot//len(datasets.id2verb)
        data_subsample = []
        if quotient == 0: ## n_shot is smaller than label space size
            chosen_cls = random.sample(list(datasets.id2label.keys()), args.n_shot)
            for cls in chosen_cls:
                data_subsample.append(random.choice(data_by_cls[cls]))
        else: ## n_shot is greater than label space size, we need to figure out remainder
            remainder = args.n_shot%len(datasets.id2verb)
            for cls in data_by_cls.keys():
                data_subsample.extend(random.sample(data_by_cls[cls], min(quotient, len(data_by_cls[cls]))))
            if remainder!=0: 
                chosen_cls = random.sample(list(datasets.id2label.keys()), remainder)
                for cls in chosen_cls:
                    data_subsample.append(random.choice(data_by_cls[cls]))
            if len(data_subsample)!=args.n_shot:
                n_leftover = args.n_shot - len(data_subsample)
                A = set(list(datasets.train_data.keys()))
                B = set(data_subsample)
                leftover_train_idx = A.difference(B)
                data_subsample.extend(random.sample(leftover_train_idx, n_leftover))
        random.shuffle(data_subsample)
    elif args.selection == "similar":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
        print("Start generate embedding")
        if datasets.args.single is False:
            train_sentences = [" ".join(t[0]) for t in datasets.train_data.values()]
            dev_sentences =  [" ".join(t[0]) for t in datasets.dev_data.values()]
        else:
            train_sentences = [t[0] for t in datasets.train_data.values()]
            dev_sentences = [t[0] for t in datasets.dev_data.values()]
        train_embedding = model.encode(train_sentences, batch_size=2)
        dev_embedding = model.encode(dev_sentences, batch_size=2)
        del model
        print("calculating cosine similarity")
        cosine_scores = np.array(util.cos_sim(dev_embedding, train_embedding))
        sim_idx = np.argsort(cosine_scores, axis=1)[:,::-1]
        demon_idx = sim_idx[:,:args.n_shot].tolist()
        demon_idx = [[list(datasets.train_data.keys())[t] for t in l] for l in demon_idx]
        data_subsample = demon_idx
        rank = [[i for i in range(args.n_shot)] for _ in range(len(demon_idx))]
        return data_subsample, rank
    elif args.selection == "pair_similar":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
        print("Start generate premise embedding")
        premise_train_sentences = [t[0][0] for t in datasets.train_data.values()]
        premise_dev_sentences =  [t[0][0] for t in datasets.dev_data.values()]
        premise_train_embedding = model.encode(premise_train_sentences, batch_size=2)
        premise_dev_embedding = model.encode(premise_dev_sentences, batch_size=2)
        print("calculating premise cosine similarity")
        premise_cosine_scores = np.array(util.cos_sim(premise_dev_embedding, premise_train_embedding))
        print("Start generate hypothesis embedding")
        hypothesis_train_sentences = [t[0][1] for t in datasets.train_data.values()]
        hypothesis_dev_sentences =  [t[0][1] for t in datasets.dev_data.values()]
        hypothesis_train_embedding = model.encode(hypothesis_train_sentences, batch_size=2)
        hypothesis_dev_embedding = model.encode(hypothesis_dev_sentences, batch_size=2)
        print("calculating premise cosine similarity")
        hypothesis_cosine_scores = np.array(util.cos_sim(hypothesis_dev_embedding, hypothesis_train_embedding))
        del model
        cosine_scores = premise_cosine_scores+hypothesis_cosine_scores
        sim_idx = np.argsort(cosine_scores, axis=1)[:,::-1]
        demon_idx = sim_idx[:,:args.n_shot].tolist()
        demon_idx = [[list(datasets.train_data.keys())[t] for t in l] for l in demon_idx]
        data_subsample = demon_idx
        rank = [[i for i in range(args.n_shot)] for _ in range(len(demon_idx))]
        return data_subsample, rank
    elif args.selection == "pair_one_similar":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
        print("Start generate embedding")
        if datasets.args.single is False:
            train_sentences = [t[0][0] for t in datasets.train_data.values()]
            dev_sentences =  [t[0][0] for t in datasets.dev_data.values()]
        train_embedding = model.encode(train_sentences, batch_size=2)
        dev_embedding = model.encode(dev_sentences, batch_size=2)
        del model
        print("calculating cosine similarity")
        cosine_scores = np.array(util.cos_sim(dev_embedding, train_embedding))
        sim_idx = np.argsort(cosine_scores, axis=1)[:,::-1]
        demon_idx = sim_idx[:,:args.n_shot].tolist()
        demon_idx = [[list(datasets.train_data.keys())[t] for t in l] for l in demon_idx]
        data_subsample = demon_idx
        rank = [[i for i in range(args.n_shot)] for _ in range(len(demon_idx))]
        return data_subsample, rank
    elif args.selection == "pair_one_similar2":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
        print("Start generate embedding")
        if datasets.args.single is False:
            train_sentences = [t[0][1] for t in datasets.train_data.values()]
            dev_sentences =  [t[0][1] for t in datasets.dev_data.values()]
        train_embedding = model.encode(train_sentences, batch_size=2)
        dev_embedding = model.encode(dev_sentences, batch_size=2)
        del model
        print("calculating cosine similarity")
        cosine_scores = np.array(util.cos_sim(dev_embedding, train_embedding))
        sim_idx = np.argsort(cosine_scores, axis=1)[:,::-1]
        demon_idx = sim_idx[:,:args.n_shot].tolist()
        demon_idx = [[list(datasets.train_data.keys())[t] for t in l] for l in demon_idx]
        data_subsample = demon_idx
        rank = [[i for i in range(args.n_shot)] for _ in range(len(demon_idx))]
        return data_subsample, rank
    elif args.selection == "cross_pair_similar":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
        print("Start generate premise embedding")
        premise_train_sentences = [t[0][0] for t in datasets.train_data.values()]
        premise_dev_sentences =  [t[0][0] for t in datasets.dev_data.values()]
        premise_train_embedding = model.encode(premise_train_sentences, batch_size=2)
        premise_dev_embedding = model.encode(premise_dev_sentences, batch_size=2)
        print("Start generate hypothesis embedding")
        hypothesis_train_sentences = [t[0][1] for t in datasets.train_data.values()]
        hypothesis_dev_sentences =  [t[0][1] for t in datasets.dev_data.values()]
        hypothesis_train_embedding = model.encode(hypothesis_train_sentences, batch_size=2)
        hypothesis_dev_embedding = model.encode(hypothesis_dev_sentences, batch_size=2)
        print("calculating premise-premise cosine similarity")
        p_p_cosine_scores = np.array(util.cos_sim(premise_dev_embedding, premise_train_embedding))
        print("calculating hypothesis-hypothesis cosine similarity")
        h_h_cosine_scores = np.array(util.cos_sim(hypothesis_dev_embedding, hypothesis_train_embedding))
        print("calculating premise-hypothesis cosine similarity")
        p_h_cosine_scores = np.array(util.cos_sim(premise_dev_embedding, hypothesis_train_embedding))
        print("calculating hypothesis-premise cosine similarity")
        h_p_cosine_scores = np.array(util.cos_sim(hypothesis_dev_embedding, premise_train_embedding))
        del model
        cosine_scores = np.max([p_p_cosine_scores, h_h_cosine_scores, p_h_cosine_scores, h_p_cosine_scores], axis=0)
        sim_idx = np.argsort(cosine_scores, axis=1)[:,::-1]
        demon_idx = sim_idx[:,:args.n_shot].tolist()
        demon_idx = [[list(datasets.train_data.keys())[t] for t in l] for l in demon_idx]
        data_subsample = demon_idx
        rank = [[i for i in range(args.n_shot)] for _ in range(len(demon_idx))]
        return data_subsample, rank
    elif args.selection == "diverse":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
        print("Start generate embedding")
        if args.data=="wnli" or args.data=="rte":
            train_sentences = [" ".join(t[0]) for t in datasets.train_data.values()]
        else:
            train_sentences = [t[0] for t in datasets.train_data.values()]
        train_embedding = model.encode(train_sentences, batch_size=2)
        del model
        clustering_models = KMeans(n_clusters=args.n_shot)
        print("fitting cluster")
        clustering_models.fit(train_embedding)
        cluster_assignment = clustering_models.labels_
        clustered_sentences = [[] for i in range(args.n_shot)]
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_sentences[cluster_id].append(list(datasets.train_data.keys())[sentence_id])
        data_subsample = [random.choice(l) for l in clustered_sentences]
        random.shuffle(data_subsample)
    elif args.selection == "zero_shot":
        data_subsample=[]
    elif args.selection == "cheat_similar":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
        print("Start generate embedding")
        if args.data=="wnli" or args.data=="rte":
            train_sentences = [" ".join(t[0]) for t in datasets.train_data.values()]
            dev_sentences =  [" ".join(t[0]) for t in datasets.dev_data.values()]
        else:
            train_sentences = [t[0] for t in datasets.train_data.values()]
            dev_sentences = [t[0] for t in datasets.dev_data.values()]
        train_embedding = model.encode(train_sentences, batch_size=2)
        dev_embedding = model.encode(dev_sentences, batch_size=2)
        del model
        print("calculating cosine similarity")
        cosine_scores = np.array(util.cos_sim(dev_embedding, train_embedding))
        sim_idx = np.argsort(cosine_scores, axis=1)[:,::-1].tolist()
        demon_idx = [[list(datasets.train_data.keys())[t] for t in l] for l in sim_idx]
        data_subsample = []
        rank = []
        for i, t in enumerate(datasets.dev_data.values()):
            dev_label = t[1]
            each_demon_idx = demon_idx[i]
            cheat_demon_idx = []
            cheat_rank = []
            for i,d in enumerate(each_demon_idx):
                if len(cheat_demon_idx)==args.n_shot:
                    break
                if datasets.train_data[d][1]==dev_label:
                    cheat_demon_idx.append(d)
                    cheat_rank.append(i)
                else:
                    continue
            data_subsample.append(cheat_demon_idx)
            rank.append(cheat_rank)
        return data_subsample, rank
    elif args.selection == "opposite_similar":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
        print("Start generate embedding")
        if args.data=="wnli" or args.data=="rte":
            train_sentences = [" ".join(t[0]) for t in datasets.train_data.values()]
            dev_sentences =  [" ".join(t[0]) for t in datasets.dev_data.values()]
        else:
            train_sentences = [t[0] for t in datasets.train_data.values()]
            dev_sentences = [t[0] for t in datasets.dev_data.values()]
        train_embedding = model.encode(train_sentences, batch_size=2)
        dev_embedding = model.encode(dev_sentences, batch_size=2)
        del model
        print("calculating cosine similarity")
        cosine_scores = np.array(util.cos_sim(dev_embedding, train_embedding))
        sim_idx = np.argsort(cosine_scores, axis=1)[:,::-1].tolist()
        demon_idx = [[list(datasets.train_data.keys())[t] for t in l] for l in sim_idx]
        data_subsample = []
        rank = []
        for i, t in enumerate(datasets.dev_data.values()):
            dev_label = t[1]
            each_demon_idx = demon_idx[i]
            opposite_demon_idx = []
            opposite_rank = []
            for i,d in enumerate(each_demon_idx):
                if len(opposite_demon_idx)==args.n_shot:
                    break
                if datasets.train_data[d][1]!=dev_label:
                    opposite_demon_idx.append(d)
                    opposite_rank.append(i)
                else:
                    continue
            data_subsample.append(opposite_demon_idx)
            rank.append(opposite_rank)
        return data_subsample, rank
    elif args.selection == "dissimilar":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
        print("Start generate embedding")
        if args.data=="wnli" or args.data=="rte":
            train_sentences = [" ".join(t[0]) for t in datasets.train_data.values()]
            dev_sentences =  [" ".join(t[0]) for t in datasets.dev_data.values()]
        else:
            train_sentences = [t[0] for t in datasets.train_data.values()]
            dev_sentences = [t[0] for t in datasets.dev_data.values()]
        train_embedding = model.encode(train_sentences, batch_size=2)
        dev_embedding = model.encode(dev_sentences, batch_size=2)
        del model
        print("calculating cosine similarity")
        cosine_scores = np.array(util.cos_sim(dev_embedding, train_embedding))
        sim_idx = np.argsort(cosine_scores, axis=1)
        demon_idx = sim_idx[:,:args.n_shot].tolist()
        demon_idx = [[list(datasets.train_data.keys())[t] for t in l] for l in demon_idx]
        data_subsample = demon_idx
        rank = [[i for i in range(len(sim_idx[0]),len(sim_idx[0])-args.n_shot, -1)] for _ in range(len(demon_idx))]
        return data_subsample, rank
    elif args.selection == "cheat_dissimilar":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
        print("Start generate embedding")
        if args.data=="wnli" or args.data=="rte":
            train_sentences = [" ".join(t[0]) for t in datasets.train_data.values()]
            dev_sentences =  [" ".join(t[0]) for t in datasets.dev_data.values()]
        else:
            train_sentences = [t[0] for t in datasets.train_data.values()]
            dev_sentences = [t[0] for t in datasets.dev_data.values()]
        train_embedding = model.encode(train_sentences, batch_size=2)
        dev_embedding = model.encode(dev_sentences, batch_size=2)
        del model
        print("calculating cosine similarity")
        cosine_scores = np.array(util.cos_sim(dev_embedding, train_embedding))
        sim_idx = np.argsort(cosine_scores, axis=1).tolist()
        demon_idx = [[list(datasets.train_data.keys())[t] for t in l] for l in sim_idx]
        data_subsample = []
        rank = []
        for i, t in enumerate(datasets.dev_data.values()):
            dev_label = t[1]
            each_demon_idx = demon_idx[i]
            cheat_demon_idx = []
            cheat_rank = []
            for i,d in enumerate(each_demon_idx):
                if len(cheat_demon_idx)==args.n_shot:
                    break
                if datasets.train_data[d][1]==dev_label:
                    cheat_demon_idx.append(d)
                    cheat_rank.append(len(sim_idx[0])-i)
                else:
                    continue
            data_subsample.append(cheat_demon_idx)
            rank.append(cheat_rank)
        return data_subsample, rank
    elif args.selection == "opposite_dissimilar":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
        print("Start generate embedding")
        if args.data=="wnli" or args.data=="rte":
            train_sentences = [" ".join(t[0]) for t in datasets.train_data.values()]
            dev_sentences =  [" ".join(t[0]) for t in datasets.dev_data.values()]
        else:
            train_sentences = [t[0] for t in datasets.train_data.values()]
            dev_sentences = [t[0] for t in datasets.dev_data.values()]
        train_embedding = model.encode(train_sentences, batch_size=2)
        dev_embedding = model.encode(dev_sentences, batch_size=2)
        del model
        print("calculating cosine similarity")
        cosine_scores = np.array(util.cos_sim(dev_embedding, train_embedding))
        sim_idx = np.argsort(cosine_scores, axis=1).tolist()
        demon_idx = [[list(datasets.train_data.keys())[t] for t in l] for l in sim_idx]
        data_subsample = []
        rank = []
        for i, t in enumerate(datasets.dev_data.values()):
            dev_label = t[1]
            each_demon_idx = demon_idx[i]
            opposite_demon_idx = []
            opposite_rank = []
            for i,d in enumerate(each_demon_idx):
                if len(opposite_demon_idx)==args.n_shot:
                    break
                if datasets.train_data[d][1]!=dev_label:
                    opposite_demon_idx.append(d)
                    opposite_rank.append(len(sim_idx[0])-i)
                else:
                    continue
            data_subsample.append(opposite_demon_idx)
            rank.append(opposite_rank)
        return data_subsample, rank
    return data_subsample

    






        

        
    

    








