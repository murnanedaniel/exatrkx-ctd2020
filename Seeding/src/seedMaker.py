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
import time
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

# Locals
from trainers import get_trainer
from utils.data_utils import load_config_dir, load_summaries, get_seed_data_loader

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
   

def load_triplets(test_loader, filelist):
    graph_dataset = test_loader.dataset
    graph_indices = np.array([g.i for g in graph_dataset])
    filelist = np.array(filelist)
    graph_names = filelist[graph_indices]
    
    return graph_dataset, graph_names

def save_triplet_hitlist(triplet_data, threshold, output_dir):
    
    e, graph_name, o = triplet_data
    
    g_ID = np.load(graph_name[:-4] + "_ID.npz", allow_pickle=True)["I"]
    triplet_IDs = np.hstack([g_ID[:,e[0,:]].T, g_ID[:,e[1,:]].T])[:,[0,1,3]]
    triplet_preds = triplet_IDs[o > threshold]
    o_preds = o[o > threshold]

    triplet_list = np.c_[triplet_preds.astype(np.int64), o_preds]
    filename = os.path.join(output_dir, os.path.splitext(os.path.basename(graph_name))[0])
    np.save(filename, triplet_list)

def get_edge_scores(result_dir, n_tasks, task):
    """
    - Takes config info for triplet training dataset (different from doublet training dataset),
    - Runs the dataset through the trained doublet network,
    - Returns edge scores with same indices as edge network input
    """

    # Load configs
    config = load_config_dir(result_dir)
    logging.info('Inferring triplets on model configuration:')
    logging.info(config)

    # Find the best epoch
    summaries = load_summaries(config)
    best_idx = summaries.valid_loss.idxmin()
    summaries.loc[[best_idx]]

    # Build the trainer and load best checkpoint
    trainer = get_trainer(output_dir=config['output_dir'], gpu=0, **config['trainer'])
    trainer.build_model(optimizer_config=config['optimizer'], **config['model'])

    best_epoch = summaries.epoch.loc[best_idx]
    trainer.load_checkpoint(checkpoint_id=best_epoch)

    logging.info("With weight system:")
    logging.info(trainer.model)
    logging.info("On device:")
    logging.info(trainer.device)

    # Load the test dataset

    test_loader, filelist = get_seed_data_loader(config, n_tasks, task)

    # Apply the model
    test_preds, test_targets = trainer.device_predict(test_loader)
    print("Graph prediction complete")

    #GET Hit ID data here and GRAPH NAMES
    graph_dataset, graph_names = load_triplets(test_loader, filelist)
     
    return test_preds, graph_dataset, graph_names

def combine_event(event_name, split_names, output_dir):
    """ Concatenates the triplet list of each subgraph """
    
    total_triplets = np.empty((0,4), dtype="int64")
    for i in np.where(split_names[:,0] == event_name)[0]:
        triplet_list = np.load(str(split_names[i,0]) + "_" + str(split_names[i,1]), allow_pickle=True)
        total_triplets = np.append(total_triplets, triplet_list, axis=0)
    triplet_filename = os.path.join(output_dir,os.path.basename(event_name))
        
    np.save(triplet_filename, total_triplets)

def process_data(output_dir, result_dir, args, threshold=0.):

    logging.info("Running inference on triplet graphs")
    
    # Calculate edge scores from best doublet model checkpoint
    edge_scores, graph_dataset, graph_names = get_edge_scores(result_dir, args.n_tasks, args.task)

    triplet_data = np.array([[gi.edge_index.numpy(), graph_name, oi.numpy()] for gi, graph_name, oi in zip(graph_dataset, graph_names, edge_scores)])
    
    logging.info("Inference complete")
    
    # SAVE TRIPLET HITLIST    
    temp_dir = os.path.join(output_dir, "temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    with mp.Pool(processes=None) as pool:
        process_fn = partial(save_triplet_hitlist, threshold=threshold, output_dir=temp_dir)
        pool.map(process_fn, triplet_data)
    
    logging.info("All files saved")
    
    if args.task == 0:
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
            process_fn = partial(combine_event, split_names = split_names, output_dir=output_dir)
            pool.map(process_fn, event_names)

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=False)

            
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

    result_dir = os.path.join(config['triplet_model_dir'], config['model_name'])
    output_dir = config['output_dir']
    threshold = config['threshold']

    process_data(output_dir, result_dir, args, threshold)

    logging.info('Processing finished')


if __name__ == '__main__':
    main()
