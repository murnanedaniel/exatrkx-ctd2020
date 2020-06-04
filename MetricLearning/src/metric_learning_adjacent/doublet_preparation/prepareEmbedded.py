"""
Data preparation script for GNN tracking.

This script processes the TrackML dataset and produces graph data on disk.
"""

# System
import os
import sys
import argparse
import logging
import multiprocessing as mp
from functools import partial
import time
import random

# Libraries
import scipy as sp

# Externals
import yaml
import numpy as np
import pandas as pd
from collections import Counter

# Locals
from .graph import SparseGraph

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('prepare.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/prepare_trackml.yaml')
    add_arg('--n-workers', type=int, default=1)
    add_arg('--task', type=int, default=0)
    add_arg('--n-tasks', type=int, default=1)
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    return parser.parse_args()

def random_rot(hits):
    """ Used for data augmentation"""
    rand_angle = np.random.rand() * 2*np.pi
    hits[:,1] = (hits[:,1] + rand_angle) % (2*np.pi)
    return hits


def augment_graph(e, y):
    
    low_edge_count = [k for k,v in Counter(np.hstack([e[0,:], e[1,:]])).items() if v < 4]
    
    new_edges = []
    for i, hit in enumerate(low_edge_count):
        random_edge = random.choice(low_edge_count[:i]+low_edge_count[i+1:])
        new_edges.append([hit, random_edge])
    
    new_edges = np.array(new_edges).T
    e = np.hstack((e, new_edges))
    y = np.hstack([y, np.zeros(new_edges.shape[1])])
    
    return e, y


def reset_hit_indices(X, e, I, pid):
    U = list(set(e[0,:]) | set(e[1,:]))
    newX = X[U]
    newI = I[U]
    newpid = pid[U]
    Reverse_U = np.zeros(len(X), dtype=int)
    Reverse_U[U] = np.arange(len(U))
    newE = np.zeros((2,e.shape[1]), dtype=int)
    newE[0,:] = Reverse_U[e[0,:]]
    newE[1,:] = Reverse_U[e[1,:]]
       
    return newX, newE, newI, newpid

def remove_duplicate_edges(X, e):
    
    # Re-introduce layer/directionality information using the r-direction
    r_mask = X[e[0,:],0] > X[e[1,:],0]
    e[0,r_mask], e[1,r_mask] = e[1,r_mask], e[0,r_mask]
    
    # Use sparse matrices to remove duplicates
    e_sparse = sp.sparse.coo_matrix(([1]*e.shape[1], e))
#     e_sparse = cpx.scipy.sparse.coo_matrix((cp.array([1]*e.shape[1]).astype('Float32'), (cp.array(e[0]).astype('Float32'), cp.array(e[1]).astype('Float32'))))
    e_sparse.sum_duplicates()
#     e_sp_sparse = sp.sparse.coo_matrix(e_sparse.get())
    
    # Remove self-edges
    e_sparse.setdiag(0)
    e_sparse.eliminate_zeros()
    
    # Reshape as numpy array
    e = np.vstack([e_sparse.row, e_sparse.col])
    
    return e

def construct_graph(X, e, y, I, pid, norm_phi_min, delta, n_phi_sections, augmented):
    """Construct one graph (e.g. from one event)"""
    """ Masks out the edges and hits falling within [phi_min, phi_min + delta_phi]"""
    
    # Mask out phi segment edges and truths, leaving full hit list
    seg_hits = (X[:,1] >= norm_phi_min) & (X[:,1] < (norm_phi_min + delta))
    seg_edges = seg_hits[e[0,:]] | seg_hits[e[1,:]] # Whether to filter by in or out end may impact training, explore this further!
    e_sec = e[:, seg_edges]
    y_sec = y[seg_edges]
       
    # Option to augment graph with random connections
    if augmented:
        e_sec, y_sec = augment_graph(e_sec, y_sec)
    
    # Fix the truth data type
    y_sec = y_sec.astype(np.float32)
    
    # Reset hit indices to avoid unused hits
    X, e_sec, I, pid = reset_hit_indices(X, e_sec, I, pid)

    # Center phi at 0
    X[:,1] = X[:,1] - norm_phi_min - (delta/2)
    # Handle case of crossing the boundary
    X[X[:,1] < (-n_phi_sections), 1] += 2*n_phi_sections
    X[X[:,1] > (n_phi_sections), 1] -= 2*n_phi_sections
    
    # Return a tuple of the results
    return SparseGraph(X, e_sec, y_sec, I, pid)

def process_event(output_dir, event_name, hits, truth, e, scores, pt_min = 0., n_phi_sections = 1, embed_feats = False, augmented=False):
    #Timing
    tic = time.time()
    
    # Calculate derived hits variables
    r = np.sqrt(hits.x**2 + hits.y**2)
    phi = np.arctan2(hits.y, hits.x)
    # Select the data columns we need
    hits = hits.assign(r=r, phi=phi)

    # Option to include embedding features
    ''' TO BE IMPLEMENTED
    if embed_feats is True:
        emb = result['hits_emb'] 
    '''
    
    feature_names = ['r', 'phi', 'z']
    feature_scale = np.array([1000., np.pi / n_phi_sections, 1000.])
    I = hits['hit_id'].to_numpy()
    X = (hits[feature_names].values / feature_scale).astype(np.float32)
    truth = truth.to_numpy()
    pt_mask = truth[:,9] > pt_min    
    e = e[:,pt_mask[e[0,:]] & pt_mask[e[1,:]]]
    
    # Remove duplicate edges
    e = remove_duplicate_edges(X, e)    
    
#     y = np.zeros(n_edges, dtype=np.float32)  ## THIS SHOULDN'T BE NECESSARY ONCE I FIX THE EMBEDDED BUILDER
    pid = truth[:,1]
    y = (pid[e[0,:]] == pid[e[1,:]]).astype(np.float32)
    

    # Construct the graph
    logging.info('Event %s, constructing graphs' % event_name)
    
    delta = 2
    
    # Loop over phi segments
    for i in range(n_phi_sections):
        
        norm_phi_min = -n_phi_sections + (i*delta)
        graph = construct_graph(X, e, y, I, pid, norm_phi_min, delta, n_phi_sections, augmented)

        # Write these graphs to the output directory
        try:
            file_name = os.path.join(output_dir, os.path.splitext(event_name)[0] + "_" + str(i))
            file_name_ID = os.path.join(output_dir, os.path.splitext(event_name)[0] + "_" + str(i) + "_ID")
        except Exception as e:
            logging.info(e)

        logging.info('Event %s, writing graph %i', event_name, i)
        np.savez(file_name, X = graph.X, e = graph.e, y = graph.y)
        np.savez(file_name_ID, I = graph.I, pid=graph.pid)
        
        logging.info('Graph (%s, %i) constructed in %.2f seconds', event_name, i, time.time() - tic)
    

def main():
    """Main function"""

    toc = time.time()
    
    # Parse the command line
    args = parse_args()

    # Setup logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)
    logging.info('Initializing')
    if args.show_config:
        logging.info('Command line config: %s' % args)

    # Load configuration
    with open(args.config) as f:
        config = yaml.load(f)
    if args.task == 0:
        logging.info('Configuration: %s' % config)

    input_dir = config['input_dir']
    all_files = os.listdir(input_dir)
    all_files.sort()
    if 'n_files' in config:
        n_files = config['n_files']
    else:
        n_files = len(all_files)
    file_prefixes = [os.path.join(input_dir, file) for file in all_files[:n_files]]
    
    # Split the input files by number of tasks and select my chunk only
    file_prefixes = np.array_split(file_prefixes, args.n_tasks)[args.task]

    # Prepare output
    output_dir = os.path.expandvars(config['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    logging.info('Writing outputs to ' + output_dir)

    # Process input files with a worker pool
    with mp.Pool(processes=args.n_workers) as pool:
        process_func = partial(process_event, output_dir=output_dir, **config['selection'])
        pool.map(process_func, file_prefixes)

    # Drop to IPython interactive shell
    if args.interactive:
        logging.info('Starting IPython interactive session')
        import IPython
        IPython.embed()

    logging.info('All complete in %i seconds', time.time() - toc)

if __name__ == '__main__':
    main()
