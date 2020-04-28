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
import plotly.graph_objects as go

# Locals
from models import get_model
import datasets.hitgraphs
from torch_geometric.data import Batch
from datasets.hitgraphs_sparse import HitGraphDataset
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import trackml.dataset

#--------------------------------- Visualisation --------------------------------------------

def draw_sample_brian_xy(hits, edges, preds, brian, pid, labels, cut=0.5, figsize=(16, 16)):
    x = hits[:,0] * np.cos(hits[:,1])
    y = hits[:,0] * np.sin(hits[:,1])
    fig, ax0 = plt.subplots(figsize=figsize)
    brian_filter = pid == brian
    p_brian_edges = edges[:,(brian_filter[edges[0,:]]) & (brian_filter[edges[1,:]]) & (preds > cut)]
    n_brian_edges = edges[:,(brian_filter[edges[0,:]]) & (brian_filter[edges[1,:]]) & (preds < cut)]
    
    # Draw the hits
    ax0.scatter(x, y, s=2, c='k', alpha=0.1)

    # Draw the segments
    for j in range(labels.shape[0]):

        # False negatives
        if preds[j] < cut and labels[j] > cut:
            ax0.plot([x[edges[0,j]], x[edges[1,j]]],
                     [y[edges[0,j]], y[edges[1,j]]],
                     '--', c='b', alpha=0.1)

        # False positives
        if preds[j] > cut and labels[j] < cut:
            ax0.plot([x[edges[0,j]], x[edges[1,j]]],
                     [y[edges[0,j]], y[edges[1,j]]],
                     '-', c='r', alpha=0.1)

        # True positives
        if preds[j] > cut and labels[j] > cut:
            ax0.plot([x[edges[0,j]], x[edges[1,j]]],
                     [y[edges[0,j]], y[edges[1,j]]],
                     '-', c='k', alpha=0.1)
   
    ax0.plot([x[p_brian_edges[0,:]], x[p_brian_edges[1,:]]], [y[p_brian_edges[0,:]], y[p_brian_edges[1,:]]], c='k', linewidth=3, alpha=0.9)
    ax0.plot([x[n_brian_edges[0,:]], x[n_brian_edges[1,:]]], [y[n_brian_edges[0,:]], y[n_brian_edges[1,:]]], c='r', linewidth=3, alpha=0.9)
    ax0.scatter(x[brian_filter], y[brian_filter], c='k', s=50, alpha=0.9)
            
    return fig, ax0

def draw_brian_triplets_xy(hits, all_edges, all_preds, brian, pid, all_labels, n_edges=1, cut=0.5, figsize=(16, 16)):
    edges, preds, labels = all_edges[:,::n_edges], all_preds[::n_edges], all_labels[::n_edges]
    
    xi, yi = [hits[:,0] * np.cos(hits[:,1]), hits[:,0] * np.sin(hits[:,1])]
    xo, yo = [hits[:,3] * np.cos(hits[:,4]), hits[:,3] * np.sin(hits[:,4])]
    fig, ax0 = plt.subplots(figsize=figsize)

    brian_filter = pid == brian
    p_brian_edges = all_edges[:,(brian_filter[all_edges[0,:]]) & (brian_filter[all_edges[1,:]]) & (all_preds > cut)]
    n_brian_edges = all_edges[:,(brian_filter[all_edges[0,:]]) & (brian_filter[all_edges[1,:]]) & (all_preds < cut)]
    
    #Draw the hits
    ax0.scatter(xi, yi, s=2, c='k')
    ax0.scatter(xo, yo, s=2, c='k')

    # Draw the segments
    for j in range(labels.shape[0]):

        # False negatives
        if preds[j] < cut and labels[j] > cut:
            ax0.plot([xi[edges[0,j]], xo[edges[0,j]]],
                     [yi[edges[0,j]], yo[edges[0,j]]],
                     '--', c='b', alpha=0.6)
            ax0.plot([xi[edges[1,j]], xo[edges[1,j]]],
                     [yi[edges[1,j]], yo[edges[1,j]]],
                     '--', c='b', alpha=0.6)

        # False positives
        if preds[j] > cut and labels[j] < cut:
            ax0.plot([xi[edges[0,j]], xo[edges[0,j]]],
                     [yi[edges[0,j]], yo[edges[0,j]]],
                     '-', c='r', alpha=0.6)
            ax0.plot([xi[edges[1,j]], xo[edges[1,j]]],
                     [yi[edges[1,j]], yo[edges[1,j]]],
                     '-', c='r', alpha=0.6)

        # True positives
        if preds[j] > cut and labels[j] > cut:
            ax0.plot([xi[edges[0,j]], xo[edges[0,j]]],
                     [yi[edges[0,j]], yo[edges[0,j]]],
                     '-', c='k', alpha=0.01)
            ax0.plot([xi[edges[1,j]], xo[edges[1,j]]],
                     [yi[edges[1,j]], yo[edges[1,j]]],
                     '-', c='k', alpha=0.01)

    ax0.plot([xi[p_brian_edges[0,:]], xi[p_brian_edges[1,:]]], [yi[p_brian_edges[0,:]], yi[p_brian_edges[1,:]]], c='y', linewidth=3, alpha=0.9)
    ax0.plot([xi[n_brian_edges[0,:]], xi[n_brian_edges[1,:]]], [yi[n_brian_edges[0,:]], yi[n_brian_edges[1,:]]], c='r', linewidth=3, alpha=0.9)
    ax0.plot([xo[p_brian_edges[0,:]], xo[p_brian_edges[1,:]]], [yo[p_brian_edges[0,:]], yo[p_brian_edges[1,:]]], c='y', linewidth=3, alpha=0.9)
    ax0.plot([xo[n_brian_edges[0,:]], xo[n_brian_edges[1,:]]], [yo[n_brian_edges[0,:]], yo[n_brian_edges[1,:]]], c='r', linewidth=3, alpha=0.9)
    ax0.scatter(xi[brian_filter], yi[brian_filter], c='k', s=50, alpha=0.9)
    ax0.scatter(xo[brian_filter], yo[brian_filter], c='k', s=50, alpha=0.9)
            
    return fig, ax0

def display_score_summary():

    scores = np.array([['Truth against truth', 'All Barrel Hits', 0.999],
                        ['DBSCAN on truth graph', 'All Barrel Hits', 0.989],
                        ['DBSCAN on 1-skipped-layer truth graph', 'All Barrel Hits', 0.986],
                        ['DBSCAN on adjacent truth graph', 'All Barrel Hits', 0.957],
    #                     ['Truth against all barrel (using hits from all doublets)', 'Doublet Graph Hits', 0.981],
                        ['Truth against all barrel (using only true doublets)', 'Doublet Graph Hits', 0.935],
                        ['DBSCAN on truth graph', 'Doublet Graph Hits', 0.907],
                        ['DBSCAN on Triplet+Doublet GNN predictions', 'Doublet Graph Hits', 0.821],
                        ['DBSCAN on Doublet GNN predictions', 'Doublet Graph Hits', 0.815],
                        ['Truth against all barrel (using only true triplets)', 'Triplet Graph Hits', 0.846],
                        ['DBSCAN on truth graph', 'Triplet Graph Hits', 0.835],
                        ['DBSCAN on Triplet GNN predictions', 'Triplet Graph Hits', 0.815],      
                        ['Triplet graph truth on narrow eta range', 'Narrow Eta', 0.912],
                        ['DBSCAN on truth graph in narrow eta range', 'Narrow Eta', 0.900],
                        ['DBSCAN on narrow eta range predictions', 'Narrow Eta', 0.876],          
                        ['Triplet graph truth on narrow eta range, plus no fragments of tracks from outside barrel', 'Narrow Eta & No Fragments', 0.925],
                        ['DBSCAN on truth graph in narrow eta range, plus no fragments of tracks from outside barrel', 'Narrow Eta & No Fragments', 0.913],
                        ['DBSCAN on narrow eta range, plus no fragments of tracks from outside barrel', 'Narrow Eta & No Fragments', 0.888],
                        ['DBSCAN on adjacent hits, at least 5 layers', 'Adjacent hits', 0.932],    
                        ['DBSCAN on adjacent hits, at least 4 layers', 'Adjacent hits', 0.916]])
                        

    # Use the hovertext kw argument for hover text
    fig = go.Figure(data=[go.Bar(x=scores[:,1], y=scores[:,2],
                hovertext=scores[:,0])], layout=go.Layout(barmode='overlay'))
    # Customize aspect
    fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5, opacity=0.6)
    fig.update_layout(title_text='GNN Performance for TrackML Score')
    fig.update_layout(yaxis_type="log")
    fig.show()
    
#------------------------------------------ Event Handling ------------------------------------

def recombine_doublet_graphs(split_names, graph_dataset, test_preds, n_phi_segments):
    
    event_names = np.unique(split_names[:,0])
    
    for event_name in event_names[1:]:
        total_e = np.empty((2,0), dtype="int64")
        total_o = np.empty(0, dtype="float64")
        total_hid = np.empty(0, dtype="int64")
        total_pid = np.empty(0, dtype="int64")
        total_X = np.empty((0,3), dtype="float64")

        delta = 2 #By the normalisation process

        for i in np.where(split_names[:,0] == event_name)[0]:

            X = graph_dataset[i].x.numpy()
            X[:,1] = X[:,1] - n_phi_segments + 1 + delta*int(split_names[i,1]) #Is this right??
            X[X[:,1] < (-n_phi_segments), 1] += 2*n_phi_segments
            X[X[:,1] > n_phi_segments, 1] -= 2*n_phi_segments
            X[:,1] = X[:,1] / n_phi_segments # Renormalise

            e = graph_dataset[i].edge_index.numpy()
            scores = test_preds[i].numpy()

            hid = np.load(split_names[i,0] + "_" + split_names[i,1] + "_ID.npz", allow_pickle=True)["I"]
            pid = np.load(split_names[i,0] + "_" + split_names[i,1] + "_ID.npz", allow_pickle=True)["pid"]

            total_e = np.append(total_e, e + len(total_hid), axis=1)
            total_o = np.append(total_o, scores)
            total_hid = np.append(total_hid, hid)
            total_pid = np.append(total_pid, pid)
            total_X = np.vstack([total_X, X])

    return total_X, total_e, total_o, total_hid, total_pid

def recombine_triplet_graphs(split_names, graph_dataset, test_preds, n_phi_segments):
    
    for file_base in np.unique(split_names[:,0]):
        total_e = np.empty((2,0), dtype="int64")
        total_o = np.empty(0, dtype="float64")
        total_hid = np.empty((2,0), dtype="int64")
        total_pid = np.empty((1,0), dtype="int64")
        total_X = np.empty((0,7), dtype="float64")
        for i in np.where(split_names[:,0] == file_base)[0]:
            e_trip = graph_dataset[i].edge_index.numpy()
            scores = test_preds[i].numpy()
            hid = np.load(split_names[i,0] + "_" + split_names[i,1] + "_ID.npz", allow_pickle=True)["I"]
            pid = np.load(split_names[i,0] + "_" + split_names[i,1] + "_ID.npz", allow_pickle=True)["pid"]
            total_e = np.append(total_e, e_trip + total_hid.shape[1], axis=1)
            total_o = np.append(total_o, scores)
            total_hid = np.append(total_hid, hid, axis=1)
            total_pid = np.append(total_pid, pid)

            X = graph_dataset[i].x.numpy()
            X[:,1] = X[:,1] - n_phi_segments + 1 + delta*int(split_names[i,1]) #Is this right??
            X[X[:,1] < (-n_phi_segments), 1] += 2*n_phi_segments
            X[X[:,1] > n_phi_segments, 1] -= 2*n_phi_segments
            X[:,1] = X[:,1] / n_phi_segments # Renormalise
            X[:,4] = X[:,4] - n_phi_segments + 1 + delta*int(split_names[i,1]) #Is this right??
            X[X[:,4] < (-n_phi_segments), 1] += 2*n_phi_segments
            X[X[:,4] > n_phi_segments, 1] -= 2*n_phi_segments
            X[:,4] = X[:,4] / n_phi_segments # Renormalise

            total_X = np.vstack([total_X, graph_dataset[i].x.numpy()])
        
    return total_X, total_e, total_o, total_hid, total_pid
        

#------------------------------------------ CLUSTERING UTILITIES -------------------------------------------------

def invert_sparse_matrix(e, o, hid, cut=0.0):
    
    e_coo = sp.sparse.coo_matrix((o[o > cut], e[:,o > cut]), shape=(len(hid), len(hid)))
    dok=sparse.dok_matrix((e_coo.shape),dtype=e_coo.dtype)
    dok._update(zip(zip(e_coo.row, e_coo.col), e_coo.data))
    e_coo = dok.tocoo()

    e_inv = e_coo
    e_inv.data = 1 - e_inv.data
    e_inv_bi = sp.sparse.coo_matrix((np.hstack([e_inv.tocoo().data, e_inv.tocoo().data]), 
                                              np.hstack([np.vstack([e_inv.tocoo().row,
                                                                    e_inv.tocoo().col]),
                                                         np.vstack([e_inv.tocoo().col, 
                                                                    e_inv.tocoo().row])])))
    return e_inv_bi

def score_and_cluster(e_inv_bi, eps=0.2):
    
    clustering = DBSCAN(eps=eps, metric="precomputed", min_samples=1).fit_predict(e_inv_bi)
    
    track_list = np.vstack([np.unique(e_inv_bi.row), clustering[np.unique(e_inv_bi.row)]])
    track_list = pd.DataFrame(track_list.T)
    track_list.columns = ["hit_id", "track_id"]
    
    return score_event(hits, track_list), track_list
    
    
def generate_truth_graph(X, hits, e, o):
    
    # Create empty edges list
    e = np.empty((0,2), dtype=np.int64)
    o = np.empty((0,1))
    true_score = 0.9
    fake_score = 0.1
    missing_over1_layer_hit = 0
    missing_only1_layer_hit = 0
    missing_only2_layer_hit = 0
    missing_only3_layer_hit = 0
    missing1_hit_dist = np.zeros(10)
    missing2_hit_dist = np.zeros(10)
    missing3_hit_dist = np.zeros(10)
    # Iterate through each layer
    for i in hits.layer.unique():
        # Iterate through each hit in the layer
        for hit in np.where(X[:,4]==i)[0]:
    #         print(hit)
            # Iterate through EACH matching PID hit in the proceeding layer, and form a doublet combo
            first_match_list = np.where((X[:,4] == (i+1)) & (X[:,7] == X[hit,7]))[0]
            second_match_list = np.where((X[:,4] == (i+2)) & (X[:,7] == X[hit,7]))[0]
            third_match_list = np.where((X[:,4] == (i+3)) & (X[:,7] == X[hit,7]))[0]
            fourth_match_list = np.where((X[:,4] == (i+4)) & (X[:,7] == X[hit,7]))[0]

            if (len(first_match_list) is 0):
                missing_over1_layer_hit += 1

            if (len(first_match_list) is 0) and (len(second_match_list)>0): 
                missing_only1_layer_hit += 1
                missing1_hit_dist[i] += 1
            if (len(first_match_list) is 0) and (len(second_match_list) is 0) and (len(third_match_list)>0): 
                missing_only2_layer_hit += 1
                missing2_hit_dist[i] += 1
            if (len(first_match_list) is 0) and (len(second_match_list) is 0) and (len(third_match_list) is 0) and len(fourth_match_list) > 0: 
                missing_only3_layer_hit += 1
                missing3_hit_dist[i] += 1

            for match in first_match_list:
                # Append to edges list
    #             print(e.shape, np.array([hit[1]["hit_id"], match[1]["hit_id"]]).shape)
                e = np.vstack([e, np.array([hit, match]).T])
                # Assign a score of 0.9
                o = np.append(o, true_score)
                # Randomly pick ANY other hit and append, assigning a score of 0.1
                e = np.vstack([e, np.array([hit, np.random.choice(np.where((X[:,4]==(i+1)) & (X[:,7] != X[hit,7]))[0])])])
                o = np.append(o, fake_score)

    #         if len(first_match_list) is 0:
    #             for match in second_match_list:
    #                 # Append to edges list
    #                 e = np.vstack([e, np.array([hit, match]).T])
    #                 # Assign a score of 0.9
    #                 o = np.append(o, true_score)
    #                 # Randomly pick ANY other hit and append, assigning a score of 0.1
    #                 e = np.vstack([e, np.array([hit, np.random.choice(np.where((X[:,4]==(i+2)) & (X[:,7] != X[hit,7]))[0])])])
    #                 o = np.append(o, fake_score)

    #         if (len(first_match_list) is 0) and (len(second_match_list) is 0):
    #             for match in third_match_list:
    #                 # Append to edges list
    #                 e = np.vstack([e, np.array([hit, match]).T])
    #                 # Assign a score of 0.9
    #                 o = np.append(o, true_score)
    # #                 Randomly pick ANY other hit and append, assigning a score of 0.1
    #                 e = np.vstack([e, np.array([hit, np.random.choice(np.where((X[:,4]==(i+3)) & (X[:,7] != X[hit,7]))[0])])])
    #                 o = np.append(o, fake_score)

    #         if (len(first_match_list) is 0) and (len(second_match_list) is 0) and (len(third_match_list) is 0):
    #             for match in fourth_match_list:
    #                 # Append to edges list
    #                 e = np.vstack([e, np.array([hit, match]).T])
    #                 # Assign a score of 0.9
    #                 o = np.append(o, true_score)
    # #                 Randomly pick ANY other hit and append, assigning a score of 0.1
    #                 e = np.vstack([e, np.array([hit, np.random.choice(np.where(X[:,4]==(i+4))[0])])])
    #                 o = np.append(o, fake_score)

    e = e.T
    
    return e