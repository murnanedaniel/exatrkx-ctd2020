"""
THIS PREPARATION SCRIPT IS TO GET A DATASET READY FOR TRIPLET TRAINING
"""

# System
import os
import sys
from pprint import pprint as pp
import argparse
import logging
import multiprocessing as mp
from functools import partial

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

# Locals
from datasets.graph import SparseGraph, save_graph
from datasets import get_data_loaders
from trainers import get_trainer
from utils.data_utils import load_config_dir, load_summaries, get_data_loader, get_IDs

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('prepareTriplets.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/prep_tripgnn.yaml')
    add_arg('--n-workers', type=int, default=1)
    add_arg('--task', type=int, default=0)
    add_arg('--n-tasks', type=int, default=1)
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    return parser.parse_args()



""" OVERALL STRUCTURE
Doublet preparation script makes graphs of (X, Ri, Ro, y, pid) -> Doublet trainer loads graphs with (X, edge_index, y)
-> Doublets trained on (X, e, y) -> Triplet preparation reads (X, Ri, Ro, y) -> Runs through doublet classifier
-> Makes graph of ([Xi,Xo,edge_score], triplet_e, y) and saves -> Triplet trainer loads graph -> Triplets trained
"""


def load_pid(filename):
    return np.load(filename)["pid"]

def get_edge_scores(result_dir, n_tasks, task):
    """
    - Takes config info for triplet training dataset (different from doublet training dataset),
    - Runs the dataset through the trained doublet network,
    - Returns edge scores with same indices as edge network input
    """

    # Load configs
    config = load_config_dir(result_dir)
    logging.info('Training doublets on model configuration:')
    logging.info(config)

    # Find the best epoch
    summaries = load_summaries(config)
    best_idx = summaries.valid_loss.idxmin()
    summaries.loc[[best_idx]]

    # Build the trainer and load best checkpoint
    task_gpu = 0
    trainer = get_trainer(output_dir=config['output_dir'], gpu=task_gpu, **config['trainer'])
    trainer.build_model(optimizer_config=config['optimizer'], **config['model'])

    best_epoch = summaries.epoch.loc[best_idx]
    trainer.load_checkpoint(checkpoint_id=best_epoch)

    logging.info("With weight system:")
    logging.info(trainer.model)
    logging.info("On device:")
    logging.info(trainer.device)

    # Load the test dataset

    test_loader = get_data_loader(config, n_tasks, task)

    # Apply the model
    test_preds, test_targets = trainer.device_predict(test_loader)
    print("Graph prediction complete")
#     test_preds, test_targets = [a.cpu() for a in test_preds], [a.cpu() for a in test_targets]
    doublet_data = test_loader.dataset
    ID_data, eventnames = get_IDs(config, n_tasks, task)

    return test_preds, doublet_data, ID_data, eventnames

def edge_to_triplet_cupy(e):
    
    e_length = e.shape[1]
    e_coo = cpx.scipy.sparse.coo_matrix((cp.array([1]*e_length).astype('Float32'), (cp.array(e[0]).astype('Float32'), cp.array(e[1]).astype('Float32'))))
    
    e_ones = cp.array([1]*e_length).astype('Float32')

    e_in_coo = cp.sparse.coo_matrix((e_ones, (e_coo.row.astype('Float32'), cp.arange(e_length).astype('Float32'))), shape=(e.max()+1,e_length))
    e_in_csr = e_in_coo.tocsr()
    e_out_coo = cp.sparse.coo_matrix((e_ones, (e_coo.col.astype('Float32'), cp.arange(e_length).astype('Float32'))), shape=(e.max()+1,e_length))
    e_out_csr = e_out_coo.tocsr()
    
    e_total = e_out_csr.T * e_in_csr
    e_total_coo = e_total.tocoo()
    e_nonzero = np.vstack([e_total_coo.row.get(), e_total_coo.col.get()])
    e_nonzero = np.asarray(e_nonzero).astype(np.int64)
    
    return e_nonzero

def edge_to_triplet(start, end, n_edges, n_hits):
    """
    An efficient algorithm to convert between an edge matrix and a triplet matrix
    """
    Ri = np.zeros((n_hits+1, n_edges))
    Ro = np.zeros((n_hits+1, n_edges))
    Ri[start, np.arange(n_edges)]=1
    Ro[end, np.arange(n_edges)]=1
    Riwhere = [np.nonzero(t)[0] for t in Ri]
    Rowhere = [np.nonzero(t)[0] for t in Ro]
    E = [np.stack(np.meshgrid(j, i),-1).reshape(-1,2) for i,j in zip(Riwhere, Rowhere)]
    return np.concatenate(E).T

def construct_triplet_graph(x, e, y, I, pid, o, include_scores, threshold):
    """
    Very similar to doublet graph builder. May take some pruning parameters.
    Takes output from doublet network.
    """
    
    # Enforce doublet score threshold
    if threshold > 0:
        e, o, y = e[:, o > threshold], o[o > threshold], y[o > threshold]

    # Remove self-edges
    self_edge_mask = e[0,:] == e[1,:]
    e, o, y = e[:,~self_edge_mask], o[~self_edge_mask], y[~self_edge_mask]
    
    # Build triplet edge index matrix
    triplet_index = edge_to_triplet_cupy(e)
    n_triplets = triplet_index.shape[1]

    # Concatenate features by edge index
    if include_scores:
        triplet_X = np.concatenate([x[e[0]],x[e[1]],np.array([o]).T], axis=1)
    else:
        triplet_X = np.concatenate([x[e[0]],x[e[1]]], axis=1)
        
    # Ground truth vector from THREE matching pids in the triplet edge
    doublet_pid = (pid[e[0,:]] == pid[e[1,:]]) * pid[e[0,:]] # FUTURE-PROOFING
    triplet_y = np.zeros(n_triplets, dtype=np.float32)
    triplet_y[:] = (doublet_pid[triplet_index[0]] == doublet_pid[triplet_index[1]]) * (doublet_pid[triplet_index[0]] != 0) # FUTURE-PROOFING
#     triplet_y[:] = y[triplet_index[0]] & y[triplet_index[1]]
    
    doublet_I = np.vstack([I[e[0,:]], I[e[1,:]]])

#     return Graph(triplet_X, triplet_Ri, triplet_Ro, triplet_y)
    return SparseGraph(triplet_X, triplet_index, triplet_y, doublet_I, doublet_pid)

def process_event(data_row, output_dir, include_scores, threshold):
    """ Handles all events, returns nothing. As in doublet case"""

    x, e, y, o, I, pid, filename = data_row
    
    logging.info("Constructing graph " + str(filename))
    graph = construct_triplet_graph(x, e, y, I, pid, o, include_scores, threshold)
    
    logging.info("Saving graph " + str(filename))
    file_name = os.path.join(output_dir, str(filename))
    file_name_ID = os.path.join(output_dir, str(filename) + "_ID")
    
    np.savez(file_name, X = graph.X, e = graph.e, y = graph.y)
    np.savez(file_name_ID, I = graph.I, pid=graph.pid)

def process_data(output_dir, result_dir, args, include_scores=True, threshold=0):

    logging.info("Processing result data")

    # Calculate edge scores from best doublet model checkpoint
    edge_scores, doublet_data, ID_data, eventnames = get_edge_scores(result_dir, args.n_tasks, args.task)
    all_data = np.array([[gi.x.numpy(), gi.edge_index.numpy(), gi.y.numpy(), oi.numpy(), ID[0], ID[1]]
                    for gi, oi, ID in zip(doublet_data, edge_scores, ID_data)])
    
#     logging.info("Data shape:")
#     logging.info(all_data.shape)
#     logging.info(all_data)
#     logging.info("Eventnames shape:")
#     logging.info(eventnames.shape)
#     logging.info(eventnames)
    
    # Attach the event ID to each graph for saving
    all_data = np.c_[all_data, eventnames.T]
    print(all_data.shape)
    logging.info("Data processed")
    
    # Process events with pool
    n_workers = args.n_workers
    
#     with mp.Pool(processes=1) as pool:
#         process_fn = partial(process_event, output_dir=output_dir, include_scores=include_scores, threshold=threshold)
#         pool.map(process_fn, all_data)

    for data_row in all_data:
        process_event(data_row, output_dir, include_scores, threshold)


def main():
    """ Main function """

    # Parse args
    args = parse_args()

    # Setup logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)
    logging.info('Initialising')
    if args.show_config:
        logging.info('Command line config: %s' % config)

    # Load config
    with open(args.config) as f:
        config = yaml.load(f)
    if args.task == 0:
        logging.info('Configuration: %s' % config)

    result_dir = config['doublet_model_dir']
    output_dir = config['output_dir']
    include_scores = config['include_scores']
    threshold = config['threshold']

    process_data(output_dir, result_dir, args, include_scores, threshold)

    logging.info('Processing finished')


if __name__ == '__main__':
    main()
