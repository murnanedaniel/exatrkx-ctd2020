"""
This file contains some common helper code for the analysis notebooks.
"""

# System
import os
import yaml
import pickle
from collections import namedtuple

# Externals
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
from torch.utils.data import Subset, DataLoader
from IPython.display import clear_output
from IPython.display import HTML, display

# Locals
from models import get_model
import datasets.hitgraphs
from torch_geometric.data import Batch
from datasets.hitgraphs_sparse import HitGraphDataset
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import trackml.dataset

from prepare import calc_eta, calc_dphi, split_detector_sections, select_hits, select_segments
from prepareEmbedded import remove_duplicate_edges, reset_hit_indices, augment_graph

##------------------------------ Visualisation ------------------------

def progress(value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))

def plot_brian(event, feature_scale, brian, phi_min = -np.pi, phi_max = np.pi, r_min = 0, r_max = 1200):
    X = event.x.numpy() * feature_scale
    X_filter = (X[:,1] > phi_min) & (X[:,1] < phi_max) & (X[:,0] >= r_min) & (X[:,0] <= r_max)
    brian_filter = event.pid.numpy() == brian
    
    x = X[:,0] * np.cos(X[:,1])
    y = X[:,0] * np.sin(X[:,1])
    
    plt.figure(figsize=(10,10))
    e = event.edge_index.numpy()

    brian_edges = e[:,(X_filter[e[0,:]]) & (X_filter[e[1,:]]) & (brian_filter[e[0,:]]) & (brian_filter[e[1,:]])]
    print(brian_edges)
    e = e[:,(X_filter[e[0,:]]) & (X_filter[e[1,:]])]
    X_ind = np.arange(len(X))
    e_filter = (np.isin(X_ind, e[0,:])) | (np.isin(X_ind, e[1,:]))
        
    plt.plot([x[e[0,:]], x[e[1,:]]], [y[e[0,:]], y[e[1,:]]], c='b', alpha=0.01)    
    plt.plot([x[brian_edges[0,:]], x[brian_edges[1,:]]], [y[brian_edges[0,:]], y[brian_edges[1,:]]], c='r', alpha=0.9)
    
    plt.scatter(x[X_filter & e_filter & ~brian_filter], y[X_filter & e_filter & ~brian_filter], c='k', alpha=0.01)
    plt.scatter(x[X_filter & brian_filter], y[X_filter & brian_filter], c='r', s=50, alpha=0.9)

##------------------------ Heuristic construction methods -----------------------

def hits_of_interest(hits, truth):
    pt_min = 0
    
    vlids = [(8,2), (8,4), (8,6), (8,8),
                 (13,2), (13,4), (13,6), (13,8),
                 (17,2), (17,4)]
    n_det_layers = len(vlids)
    # Select barrel layers and assign convenient layer number [0-9]
    vlid_groups = hits.groupby(['volume_id', 'layer_id'])
    hits = pd.concat([vlid_groups.get_group(vlids[i]).assign(layer=i)
                      for i in range(n_det_layers)])
    
    # Calculate PER-HIT particle transverse momentum
    pt = np.sqrt(truth.tpx**2 + truth.tpy**2)
    truth = truth.assign(pt=pt)
    # Calculate derived hits variables
    r = np.sqrt(hits.x**2 + hits.y**2)
    phi = np.arctan2(hits.y, hits.x)
    # Select the data columns we need
    hits = (hits[['hit_id', 'x', 'y', 'z', 'layer']]
            .assign(r=r, phi=phi)
            .merge(truth[['hit_id', 'particle_id', 'pt']], on='hit_id'))
    
    return hits

def construct_graph(hits, layer_pairs,
                              phi_slope_max, z0_max,
                              feature_names,
                              feature_scale):
    layer_groups = hits.groupby('layer')
    segments = []
    for (layer1, layer2) in layer_pairs:
        # Find and join all hit pairs
        try:
            hits1 = layer_groups.get_group(layer1)
            hits2 = layer_groups.get_group(layer2)
        # If an event has no hits on a layer, we get a KeyError.
        # In that case we just skip to the next layer pair
        except KeyError as e:
            logging.info('skipping empty layer: %s' % e)
            continue
        # Construct the segments
        segments.append(select_segments(hits1, hits2, phi_slope_max, z0_max))
        # Combine segments from all layer pairs
    segments = pd.concat(segments)
    
#     print("Segments selected in", event_file[-4:])
    
    X = (hits[feature_names].values / feature_scale).astype(np.float32)
    pid = (hits['particle_id'].values).astype(np.int64)
    I = (hits['hit_id'].values).astype(np.int64)
    n_edges = len(segments)
    n_hits = len(hits)
    
    pid1 = hits.particle_id.loc[segments.index_1].values
    pid2 = hits.particle_id.loc[segments.index_2].values
    y = np.zeros(n_edges, dtype=np.float32)
    y[:] = (pid1 == pid2)
    
    hit_idx = pd.Series(np.arange(n_hits), index=hits.index)
    seg_start = hit_idx.loc[segments.index_1].values
    seg_end = hit_idx.loc[segments.index_2].values
    
    e = np.vstack([seg_start, seg_end])
    
    data = Data(x = torch.from_numpy(X).float(), edge_index = torch.from_numpy(e), y = torch.from_numpy(y), I = torch.from_numpy(I), pid=torch.from_numpy(pid))
    
    return data

def build_event(event_file, pt_min, phi_slope_max, z0_max, feature_names, feature_scale, n_phi_sections=1, n_eta_sections=1):
    hits, particles, truth = trackml.dataset.load_event(
        event_file, parts=['hits', 'particles', 'truth'])
    hits = select_hits(hits, truth, particles, pt_min=pt_min).assign(evtid=int(event_file[-9:]))
    
    phi_range, eta_range = [-np.pi, np.pi], [-5, 5]
    phi_edges = np.linspace(*phi_range, num=n_phi_sections+1)
    eta_edges = np.linspace(*eta_range, num=n_eta_sections+1)
    hits_sections = split_detector_sections(hits, phi_edges, eta_edges)
    
    # Define adjacent layers
    n_det_layers = 10
    l = np.arange(n_det_layers)
    layer_pairs = np.stack([l[:-1], l[1:]], axis=1)
    
    graphs_all = [construct_graph(section_hits, layer_pairs=layer_pairs,
                              phi_slope_max=phi_slope_max, z0_max=z0_max,
                              feature_names=feature_names,
                              feature_scale=feature_scale)
                              for section_hits in hits_sections]
    
    return graphs_all

def prepare_event(event_file, pt_min, phi_slope_max, z0_max, feature_names, feature_scale, n_phi_sections=1, iter=None, num_samples=None, out=None):
    
    graphs_all = build_event(event_file, pt_min, phi_slope_max, z0_max, feature_names, feature_scale, n_phi_sections)

    if iter is not None and num_samples is not None:
        out.update(progress(iter, num_samples))    

    return graphs_all

##------------------------------------ Embedding construction methods ----------------------------


def construct_AE_graph(X, e, y, I, pid, norm_phi_min, delta, n_phi_sections, augmented, fully_connected):
    """Construct one graph (e.g. from one event)"""
    """ Masks out the edges and hits falling within [phi_min, phi_min + delta_phi]"""
    
    # Mask out phi segment edges and truths, leaving full hit list
    seg_hits = (X[:,1] >= norm_phi_min) & (X[:,1] < (norm_phi_min + delta))
    if fully_connected:
        seg_edges = seg_hits[e[1,:]] | seg_hits[e[0,:]]
    else:
        seg_edges = seg_hits[e[1,:]] 
        
    e_sec = e[:, seg_edges]
    y_sec = y[seg_edges]
    
    # Prepare the features data types    
    y_sec = y_sec.astype(np.float32)
    
    # Option to augment graph with random connections
    if augmented:
        e_sec, y_sec = augment_graph(e_sec, y_sec)
    
    # Reset hit indices to avoid unused hits
    X, e_sec, I, pid = reset_hit_indices(X, e_sec, I, pid)

    # Center phi at 0
    X[:,1] = X[:,1] - norm_phi_min - (delta/2)
    # Handle case of crossing the boundary
    X[X[:,1] < (-n_phi_sections), 1] += 2*n_phi_sections
    X[X[:,1] > (n_phi_sections), 1] -= 2*n_phi_sections
    
    
    graph = Data(x = torch.from_numpy(X).float(), edge_index = torch.from_numpy(e_sec), y = torch.from_numpy(y_sec), I = torch.from_numpy(I), pid = torch.from_numpy(pid))
    
    # Return a tuple of the results
    return graph

def prepare_AE_event(event, feature_names, feature_scale, pt_min, n_phi_sections=1, iter=None, num_samples=None, out=None, augmented=False, fully_connected=False):
    
    event = np.load(event, allow_pickle=True)
    
    hits, truth, e, scores = event['hits'], event['truth'].reset_index(drop=True), event['neighbors'], event['scores']
    
    # Calculate derived hits variables
    r = np.sqrt(hits.x**2 + hits.y**2)
    phi = np.arctan2(hits.y, hits.x)
    # Select the data columns we need
    hits = hits.assign(r=r, phi=phi)
    
    I = hits['hit_id'].to_numpy().astype(np.float32)
    X = (hits[feature_names].values / feature_scale).astype(np.float32)
    truth = truth.to_numpy()
    pt_mask = truth[:,9] > pt_min
    e = e[:,pt_mask[e[0,:]] & pt_mask[e[1,:]]]
    
    # Remove duplicate edges
    e = remove_duplicate_edges(X, e)
    
    pid = truth[:,1]
    y = (pid[e[0,:]] == pid[e[1,:]]).astype(np.float32)
    
    data = []
    
    delta = 2
    
    for i in range(n_phi_sections):
        
        norm_phi_min = -n_phi_sections + (i*delta)
        graph = construct_AE_graph(X, e, y, I, pid,  norm_phi_min, delta, n_phi_sections, augmented=augmented, fully_connected=fully_connected)
        data.append(graph)
    
    
    if iter is not None and num_samples is not None:
        out.update(progress(iter, num_samples))    
    
    return data