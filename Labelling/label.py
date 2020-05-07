# System
import os
import sys
from pprint import pprint as pp
import argparse
import logging
import multiprocessing as mp
from functools import partial
from time import time
import shutil

# Externals
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_scatter import scatter_add
import cupy as cp
import cupyx as cpx
import scipy as sp

from sklearn.cluster import DBSCAN

# Locals
# sys.path.append('GraphLearning/src')
from GraphLearning.src.trainers import get_trainer
from Seeding.src.utils.data_utils import load_config_dir, load_summaries, get_seed_data_loader

if torch.cuda.is_available():
    DEVICE='cuda'
else:
    DEVICE='cpu'

    
def load_triplets(test_loader, filelist):
    graph_dataset = test_loader.dataset
    graph_indices = np.array([g.i for g in graph_dataset])
    filelist = np.array(filelist)
    graph_names = filelist[graph_indices]
    
    return graph_dataset, graph_names

def save_triplet_hitlist(triplet_data, threshold, output_dir):
    
    e, graph_name, o = triplet_data
    
    g_ID = np.load(graph_name[:-4] + "_ID.npz", allow_pickle=True)["I"]
    
    triplet_preds = np.hstack([g_ID[:,e[0,o > threshold]], g_ID[:,e[1,o > threshold]]]).T
    
#     triplet_IDs = np.hstack([g_ID[:,e[0,:]].T, g_ID[:,e[1,:]].T])[:,[0,1,3]]
#     triplet_preds = triplet_IDs[o > threshold]
    o_preds = np.hstack([o[o > threshold], o[o > threshold]]).T

#     print(triplet_preds.shape, o_preds.shape)
    
    triplet_list = np.c_[triplet_preds.astype(np.int64), o_preds]
    
    filename = os.path.join(output_dir, os.path.splitext(os.path.basename(graph_name))[0])
    np.save(filename, triplet_list)

def get_edge_scores(load_path, triplet_artifacts, n_tasks, task):
    """
    - Takes config info for triplet training dataset (different from doublet training dataset),
    - Runs the dataset through the trained doublet network,
    - Returns edge scores with same indices as edge network input
    """

    # Load configs
    config = load_config_dir(triplet_artifacts)
    logging.info('Inferring triplets on model configuration:')
    logging.info(config)

    # Find the best epoch
    summaries = load_summaries(config)
    best_idx = summaries.valid_loss.idxmin()
    summaries.loc[[best_idx]]

    # Build the trainer and load best checkpoint
    task_gpu = 0 if DEVICE=='cuda' else None
    trainer = get_trainer(output_dir=config['output_dir'], gpu=task_gpu, **config['trainer'])
    trainer.build_model(optimizer_config=config['optimizer'], **config['model'])

    best_epoch = summaries.epoch.loc[best_idx]
    trainer.load_checkpoint(checkpoint_id=best_epoch)

    logging.info("With weight system:")
    logging.info(trainer.model)
    logging.info("On device:")
    logging.info(trainer.device)

    # Load the test dataset

    test_loader, filelist = get_seed_data_loader(load_path, n_tasks, task)

    # Apply the model
    test_preds, test_targets = trainer.device_predict(test_loader)
    print("Graph prediction complete")

    #GET Hit ID data here and GRAPH NAMES
    graph_dataset, graph_names = load_triplets(test_loader, filelist)
     
    return test_preds, graph_dataset, graph_names

def combine_event(event_name, split_names):
    """ Concatenates the triplet list of each subgraph """
    
    total_triplets = np.empty((0,3))
    for i in np.where(split_names[:,0] == event_name)[0]:
        triplet_list = np.load(str(split_names[i,0]) + "_" + str(split_names[i,1]), allow_pickle=True)
        total_triplets = np.append(total_triplets, triplet_list, axis=0)
        
    return total_triplets

def cluster(e_csr_bi, epsilon):
    
    clustering = DBSCAN(eps=epsilon, metric="precomputed", min_samples=1).fit_predict(e_csr_bi)
    track_labels = np.vstack([np.unique(e_csr_bi.tocoo().row), clustering[np.unique(e_csr_bi.tocoo().row)]])
    track_labels = pd.DataFrame(track_labels.T)
    track_labels.columns = ["hit_id", "track_id"]
    
    # Add TrackML scoring here and print
    
    return track_labels
    

def convert_to_bidirectional(e_csr):
    
    # Invert to treat score as an inverse distance
    e_csr.data = 1 - e_csr.data
    e_csr_bi = sp.sparse.coo_matrix((np.hstack([e_csr.tocoo().data, e_csr.tocoo().data]), 
                                     np.hstack([np.vstack([e_csr.tocoo().row, e_csr.tocoo().col]),                                                                   
                                                np.vstack([e_csr.tocoo().col, e_csr.tocoo().row])])))
    
    return e_csr_bi

def triplets_to_doublets(triplet_edges, triplet_scores, label_cut):
       
    
    e_doublet_coo = sp.sparse.coo_matrix((triplet_edges.max()+1, triplet_edges.max()+1))
    
    dok = sp.sparse.dok_matrix((e_doublet_coo.shape), dtype=e_doublet_coo.dtype)
    dok._update(zip(zip(triplet_edges[:,0], triplet_edges[:,1]), [1]*triplet_edges.shape[0])) # Could be converted to actual scores
    e_csr = dok.tocsr()
    
    return e_csr

def save_labels(track_labels, event_name, output_dir):
    
    label_filename = os.path.join(output_dir, event_name)
    
    np.save(label_filename, track_labels)
    

# def recombine_triplet_graphs(split_names, graph_dataset, test_preds, n_phi_segments):
    
# #     for file_base in np.unique(split_names[:,0]):
#     # Needs to load data as in combine_event()
#     total_e = np.empty((2,0), dtype="int64")
#     total_o = np.empty(0, dtype="float64")
#     total_hid = np.empty((2,0), dtype="int64")
#     total_pid = np.empty((1,0), dtype="int64")
#     total_X = np.empty((0,7), dtype="float64")
#     for i in np.where(split_names[:,0] == file_base)[0]:
#         e_trip = graph_dataset[i].edge_index.numpy()
#         scores = test_preds[i].numpy()
#         hid = np.load(split_names[i,0] + "_" + split_names[i,1] + "_ID.npz", allow_pickle=True)["I"]
#         pid = np.load(split_names[i,0] + "_" + split_names[i,1] + "_ID.npz", allow_pickle=True)["pid"]
#         total_e = np.append(total_e, e_trip + total_hid.shape[1], axis=1)
#         total_o = np.append(total_o, scores)
#         total_hid = np.append(total_hid, hid, axis=1)
#         total_pid = np.append(total_pid, pid)

#         X = graph_dataset[i].x.numpy()
#         X[:,1] = X[:,1] - n_phi_segments + 1 + delta*int(split_names[i,1]) #Is this right??
#         X[X[:,1] < (-n_phi_segments), 1] += 2*n_phi_segments
#         X[X[:,1] > n_phi_segments, 1] -= 2*n_phi_segments
#         X[:,1] = X[:,1] / n_phi_segments # Renormalise
#         X[:,4] = X[:,4] - n_phi_segments + 1 + delta*int(split_names[i,1]) #Is this right??
#         X[X[:,4] < (-n_phi_segments), 1] += 2*n_phi_segments
#         X[X[:,4] > n_phi_segments, 1] -= 2*n_phi_segments
#         X[:,4] = X[:,4] / n_phi_segments # Renormalise

#         total_X = np.vstack([total_X, graph_dataset[i].x.numpy()])
        
#     return total_X, total_e, total_o, total_hid, total_pid


def process_event(event_name, split_names, output_dir, label_cut, epsilon):
    
    # Recombine triplet graphs by loading all files in event
    total_triplets = combine_event(event_name, split_names)
    
    triplet_edges = total_triplets[:,:2].T.astype(dtype='int64')
    triplet_scores = total_triplets[:,2].T
    
    
    # Convert triplets to doublets
    e_csr = triplets_to_doublets(triplet_edges, triplet_scores, label_cut)
    
    # Cluster and produce track list
    e_csr_bi = convert_to_bidirectional(e_csr)
    
    # Save track labels
    track_labels = cluster(e_csr_bi, epsilon)
    
    save_labels(track_labels, event_name, output_dir)

    
def process_data(save_path, load_path, triplet_artifacts, label_threshold, epsilon, n_tasks, task):

    logging.info("Running inference on triplet graphs")
    
    # Calculate edge scores from best doublet model checkpoint
    edge_scores, graph_dataset, graph_names = get_edge_scores(load_path, triplet_artifacts, n_tasks, task)

    triplet_data = np.array([[gi.edge_index.numpy(), graph_name, oi.numpy()] for gi, graph_name, oi in zip(graph_dataset, graph_names, edge_scores)])
    
    logging.info("Inference complete")
    
    # SAVE TRIPLET HITLIST    
    temp_dir = os.path.join(save_path, "temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    with mp.Pool(processes=None) as pool:
        process_fn = partial(save_triplet_hitlist, threshold=label_threshold, output_dir=temp_dir)
        pool.map(process_fn, triplet_data)
    
    logging.info("All files saved")
    
    if task == 0:
        # IS THIS THE CORRECT LENGTH???
        triplet_data_length = len(triplet_data)
              
        while(len(os.listdir(temp_dir)) < triplet_data_length):
            print("Waiting")
            time.sleep(10) # Want to wait until all files a
        
        # RELOAD FILELIST AND SPLIT
        filelist = os.listdir(temp_dir)
        split_names = np.array([[os.path.join(temp_dir,file[:-6]), file[-5:]] for file in filelist])    
        event_names = np.unique(split_names[:,0])

        with mp.Pool(processes=None) as pool:
            process_fn = partial(process_event, split_names = split_names, output_dir=save_path, label_cut=label_threshold, epsilon=epsilon)
            pool.map(process_fn, event_names)
            
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=False)

            
def main(args, force=False):
    """ Main function """

    tic = time()
    
    save_path = os.path.join(args.data_storage_path, 'labels')
    load_path = os.path.join(args.data_storage_path, 'triplet_graphs')    

    os.makedirs(save_path, exist_ok=True)
    
    # Setup logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG #if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)
    logging.info('Initialising')


    process_data(save_path, load_path, args.triplet_artifacts, args.label_threshold, args.epsilon, args.n_tasks, args.task)

    logging.info('Processing finished')


if __name__ == '__main__':
    main()
