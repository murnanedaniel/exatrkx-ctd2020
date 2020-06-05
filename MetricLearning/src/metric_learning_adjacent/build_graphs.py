import os
import time
import pickle
import numpy as np
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from functools import partial

from sklearn.neighbors import KDTree

from .doublet_preparation.prepareEmbedded import process_event
from .train_embed.utils_experiment import load_model as load_embed_model
from .train_filter.utils_experiment import load_model as load_filter_model

sys.path.append('MetricLearning/src/') # Handles certain loading issues


'''
1) embed hits
2) build pairs
3) filter pairs
4) put remaining pairs in graph format
'''

if torch.cuda.is_available():
    DEVICE='cuda'
else:
    DEVICE='cpu'

#########################################################
#                   FILTER SKIP LAYERS                  #
#########################################################

ALL_LAYERS = [[8,2],
              [8,4],
              [8,6],
              [8,8],
              [13,2],
              [13,4],
              [13,6],
              [13,8],
              [17,2],
              [17,4]]

def get_which_layer(vol_id, lay_id):
    for i, (v, l) in enumerate(ALL_LAYERS):
        if (v==vol_id) and (l==lay_id):
            return i

def get_adj_layers(vol_id, lay_id):
    # which_layer = np.argwhere((np.array([vol_id, lay_id])==ALL_LAYERS).all(axis=1))[0,0]
    which_layer = get_which_layer(vol_id, lay_id)
    if which_layer == 0:
        return [ALL_LAYERS[1]]
    elif which_layer == len(ALL_LAYERS)-1:
        return [ALL_LAYERS[which_layer-1]]
    else:
        return [ALL_LAYERS[which_layer-1], ALL_LAYERS[which_layer+1]]

def filter_one_neighborhood(vol_id, lay_id, neighbors, volume_ids, layer_ids):
    adj_layers = get_adj_layers(vol_id, lay_id)
#     print(neighbors)
    neighbor_vol_id = volume_ids[neighbors].reshape(-1,1)
    neighbor_lay_id = layer_ids[neighbors].reshape(-1,1)
    vol_lay_ids = np.concatenate((neighbor_vol_id, neighbor_lay_id), axis=1)

#     print(neighbor_vol_id)
#     print(neighbor_lay_id)
#     print(vol_lay_ids)
    
    
    if len(adj_layers)==1:
        keep_neighbors = (adj_layers[0]==vol_lay_ids).all(axis=1)
#         print("1 adj_layer:", keep_neighbors)
    else:
        keep_below = (adj_layers[0]==vol_lay_ids).all(axis=1)
        keep_above = (adj_layers[1]==vol_lay_ids).all(axis=1)
        keep_neighbors = np.logical_or(keep_below, keep_above)
#         print("Many adj_layers:", keep_neighbors)

    return np.array(neighbors)[keep_neighbors]


#########################################################
#                   FILTER INFERENCE                    #
#########################################################
class EdgeData(Dataset):
    def __init__(self, hits, vol, neighbors):
        super(EdgeData, self).__init__()
        self.hits = hits
        self.neighbors = neighbors
        self.volume_id = vol[0]
        self.layer_id = vol[1]

    def __getitem__(self, index):
        neighbors = self.neighbors[index]
#         print(self.__dict__)
        neighbors = filter_one_neighborhood(self.volume_id[index],
                                            self.layer_id[index],
                                            neighbors,
                                            self.volume_id,
                                            self.layer_id)
        if len(neighbors) == 0:
            neighbors = self.neighbors[index][0:1]
        idx_a = np.array([index]*len(neighbors), dtype=int)

        hits_a = torch.FloatTensor(self.hits[idx_a])
        hits_b = torch.FloatTensor(self.hits[neighbors])

        X = torch.cat((hits_a, hits_b), axis=1)
        idx_pairs = np.stack((idx_a, neighbors), 1)
        return torch.LongTensor(idx_pairs), X

    def __len__(self):
        return len(self.neighbors)

def my_collate(samples):
    idx = [s[0] for s in samples]
    X   = [s[1] for s in samples]
    idx = torch.cat(idx, dim=0)
    X   = torch.cat(X,   dim=0)
    return idx, X

def predict_pairs(loader, filter_model, batch_size):
    idx_pairs = []
    scores = []
    with torch.autograd.no_grad():
        for i, (idx, X) in enumerate(loader):
            X = X.to(DEVICE, non_blocking=True)
            preds = filter_model(X)

            if len(preds.shape)==0:
                preds = preds.reshape(1)

            idx_pairs.append(idx)
            scores.append(preds.cpu())

            if (i%(len(loader)//5))==0:
#                 print("  {:6d}".format(i*batch_size))
                print("  {:3.0f}% of doublets filtered".format(i/len(loader)*100))

    idx_pairs = torch.cat(idx_pairs, dim=0)
    scores = torch.cat(scores)

    return idx_pairs.numpy(), scores.numpy()

def apply_filter(idx_pairs, scores, threshold):
    nb_before = len(scores)
    where_keep = np.argwhere(scores >= threshold)[:,0]
    idx_pairs = idx_pairs[where_keep]
    scores = scores[where_keep]
    nb_after = len(scores)
    return idx_pairs, scores

def filter_neighbors(hits, vol, neighbors, filter_model, threshold):
    batch_size = 64
    num_workers = 12 if DEVICE=='cuda' else 0 # The first equality may change depending on your CUDA configuration
    dataset = EdgeData(hits, vol, neighbors)
    loader = DataLoader(dataset,
                        batch_size = batch_size,
                        num_workers = num_workers,
                        collate_fn = my_collate)
    idx_pairs, scores = predict_pairs(loader, filter_model, batch_size)
#     print("   {:6.5f}% neighbors before filter".format((1.0 * len(scores)) / len(hits)))
    idx_pairs, scores = apply_filter(idx_pairs, scores, threshold)
#     print("   {:6.5f}% neighbors after filter".format( (1.0 * len(scores)) / len(hits)))
    return idx_pairs.transpose(), scores

#########################################################
#                   EMBED INFERENCE                     #
#########################################################

def get_emb_neighbors(X, emb_model, radius):
    X = torch.FloatTensor(X).to(DEVICE)
    with torch.autograd.no_grad():
        emb = emb_model(X)
    emb = emb.data.cpu().numpy()

    tree = KDTree(emb, leaf_size = 200)
    neighbors = tree.query_radius(emb, r = radius)
    return neighbors


#################################################
#                   GRAPH BUILD                 #
#################################################

def load_event(data_path, event_name):
    event_path = os.path.join(data_path, event_name)
    with open(event_path, 'rb') as f:
        hits, truth = pickle.load(f)
    return hits, truth

def save_event(save_path, event_name, hits, truth, neighbors, scores):
    event = {'hits':hits,
             'truth':truth,
             'neighbors':neighbors,
             'scores':scores}
    save_filepath = os.path.join(save_path, event_name)
    with open(save_filepath, 'wb') as f:
        pickle.dump(event, f)

def build_one_graph(event_name,
                    data_path,
                    save_path,
                    feature_names,
                    emb_model,
                    filter_model,
                    radius,
                    threshold,
                    pt_cut,
                    num_phi_sections):
    hits, truth = load_event(data_path, event_name)
    X = hits[feature_names].values
    vol = hits[['volume_id', 'layer_id']].values.T
    neighbors = get_emb_neighbors(X, emb_model, radius)
    print("Constructing neighborhood of:", event_name)
    neighbors, scores = filter_neighbors(X, vol, neighbors, filter_model, threshold)
    
    # Convert unstructured data into graph and save
    process_event(save_path, event_name, hits, truth, neighbors, scores, pt_min = pt_cut, n_phi_sections = num_phi_sections)


def main(args, force=False):
    save_path = os.path.join(args.data_storage_path, 'doublet_graphs')
    load_path = os.path.join(args.data_storage_path, 'preprocess_raw')
    os.makedirs(save_path, exist_ok=True)
    event_files = [os.path.splitext(file)[0] for file in os.listdir(load_path)]

    # Filter already existing event files
    existing_files = [os.path.splitext(file)[0].split('_')[0] for file in os.listdir(save_path)]
    if not force:
        remaining_events = list(set(event_files) - set(existing_files))
        print("%i events already constructed, %i remaining" % (len(set(existing_files)), len(remaining_events)))
    else:
        remaining_events = event_files

    # Split the files into n_tasks and select the ith split
    task_event_files = [event + ".pickle" for event in np.array_split(remaining_events, args.n_tasks)[args.task]]

    best_emb_path, best_filter_path = os.path.join(args.artifact_storage_path, 'metric_learning_emb', 'best_model.pkl'), os.path.join(args.artifact_storage_path, 'metric_learning_filter', 'best_model.pkl')    
    
    emb_model = load_embed_model(best_emb_path, DEVICE).to(DEVICE)
    filter_model = load_filter_model(best_filter_path, DEVICE).to(DEVICE)
    emb_model.eval()
    filter_model.eval()

    t0 = time.time()
    # Handle case of multiple, single-GPU distribution vs. multiple-nodes-multiple-CPUs
    if DEVICE=='cuda':
        for i, event_name in enumerate(task_event_files): 
            print("Beginning graph construction {} / {}".format(i+1, len(remaining_events)))
            build_one_graph(event_name,
                            load_path,
                            save_path,
                            args.feature_names,
                            emb_model,
                            filter_model,
                            args.emb_radius,
                            args.filter_threshold,
                            args.pt_cut,
                            args.num_phi_sections)                        

            t1 = time.time()
            print("{:5.1f} avg s / sample".format( (t1-t0) / (i+1) ))
    
    else:
        with mp.Pool() as pool:
            process_fn = partial(build_one_graph, 
                                 data_path = load_path,
                                 save_path = save_path,
                                 feature_names = args.feature_names,
                                 emb_model = emb_model,
                                 filter_model = filter_model,
                                 radius = args.emb_radius,
                                 threshold = args.filter_threshold,
                                 pt_cut = args.pt_cut,
                                 num_phi_sections = args.num_phi_sections)
            pool.map(process_fn, task_event_files)
    print("Total time:", time.time() - t0)
    
    return
