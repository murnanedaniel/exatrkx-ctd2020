import os
import sys
import yaml
import pickle
import argparse
import numpy as np
from numba import jit

import torch
from sklearn.neighbors import KDTree

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#####################################
#               UTILS               #
#####################################
def read_args():

  parser = argparse.ArgumentParser(description=
                      'Arguments for GNN model and experiment')
  add_arg = parser.add_argument

  add_arg('--model_file', help='Location of trained embed model',required=True)
  add_arg('--data_folder', help='Preprocessed data directory',required=True)
  add_arg('--save_dir', help='Directory to save sparse graph',required=True)
  add_arg('--kNN', help='Number neighbors, sparse graph',type=int,default=None)
  add_arg('--ball', help='Nbhood ball, sparse graph',type=float,default=None)
  add_arg('--loss_knn', help='# neighbors for loss fct',type=int,default=30)
  add_arg('--loss_rand', help='# random neighbors for loss fct',type=int,default=20)
  add_arg('--include_model_emb', help='Set flag to save raw emb',action='store_true')

  return parser.parse_args()

def load_event(data_dir, event_filename):
    with open(os.path.join(data_dir, event_filename), 'rb') as f:
        hits, truth = pickle.load(f)
    return hits, truth

def load_stats(norm_filepath):
    with open(norm_filepath, 'r') as f:
        norm_stats = yaml.load(f)
    return norm_stats

def save_event(hits,
               hits_emb,
               truth,
               save_dir,
               event_filename):
    save_filepath = os.path.join(save_dir, event_filename)
    event = {'hits':hits,
             'hits_emb':hits_emb,
             'truth':truth}
    with open(save_filepath, 'wb') as f:
        pickle.dump(event, f)

def add_random_edges(neighbors, nb_random):
    '''
    add a random numer of edges to the current set of neighbors
    WARNING: This method allows for duplicate edges
    '''
    nb_hits=len(neighbors)
    rand_neighbors = np.random.randint(low=0,
                                       high=nb_hits,
                                       size=(nb_hits, nb_random))
    neighbors = np.concatenate((neighbors, rand_neighbors),axis=1)
    return neighbors

#########################################################
#                   GRAPH EVALUATION                    #
#########################################################
@jit
def pred_edge_stats(pred_neighbors, true_neighbors):
    pred = pred_neighbors.tolist()
    nb_true = 0
    nb_pred = 0
    nb_found = 0
    for i, (p, t) in enumerate(zip(pred, true_neighbors)):
        nb_true += len(t)
        nb_pred += len(p)
        true_set = set(t)
        found_set = true_set.intersection(p)
        nb_found += len(found_set)
    percent_found = float(nb_found) / nb_true
    return percent_found, nb_found, nb_pred

#################################################
#                   BUILD GRAPH                 #
#################################################
def embed_one_file(model,
                   data_dir,
                   event_filename,
                   feature_names,
                   save_dir):
    hits, truth = load_event(data_dir, event_filename)
    hits_emb = emb_hits(model, hits, feature_names)

    save_event(hits,
               hits_emb,
               truth,
               save_dir,
               event_filename)

def emb_hits(model, hits, feature_names):
    hits = hits[feature_names].values
    model.eval()
    with torch.autograd.no_grad():
        hits_emb = model(torch.Tensor(hits).to(DEVICE))
    return hits_emb.cpu().data.numpy()

#############################################
#                   MAIN                    #
#############################################
def main(data_dir, save_dir, emb_model, feature_names, force=False):
    save_dir = os.path.join(save_dir, 'metric_embed')

    if os.path.isdir(save_dir) and (not force):
        print("Embed directory already exists. Not forcing")
    else:
        model = emb_model.to(DEVICE)
        event_files = os.listdir(data_dir)
        os.makedirs(save_dir, exist_ok=True)
        for i, e in enumerate(event_files):
            embed_one_file(model, data_dir, e, feature_names, save_dir)
            if (i%100) == 0:
                print("{:6d} of {}".format(i+1, len(event_files)))
    return save_dir

if __name__ == "__main__":
    main()
