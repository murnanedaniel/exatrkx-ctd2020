"""
This file contains some common helper code for the analysis notebooks.
"""

# System
import os
import yaml
import pickle
from collections import namedtuple
from IPython.display import clear_output


# Externals
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import torch.nn as nn
# from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
import torch_geometric.nn as tnn
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import *

# Locals
from torch_geometric.data import Batch
from utils.toy_utils import *

# Interactive
import ipywidgets as widgets
from ipywidgets import interact, interact_manual

#_________________________________ Dataset Generation _____________________


def gen_edge_class_dataset(**kwargs):
    dataset = [gen_edge_class(i, **kwargs) for i in range(int(kwargs["num_samples"]))]
    clear_output()
    print("Dataset generation complete")
    return dataset

# Generate event data from random parameters
def gen_edge_class(iter, **kwargs):
    clear_output(wait=True)
    
    """ I don't like doing it this way, but it makes the notebook very clean """
    event_size_min, event_size_max, curve_min, curve_max, height, num_layers, max_angle = kwargs["event_size_min"], kwargs["event_size_max"], kwargs["curve_min"], kwargs["curve_max"], kwargs["height"], kwargs["num_layers"], kwargs["max_angle"]*np.pi
    
    """ Feed params into randomiser """
    while True:
        radii, dirs, signs, event_size = rand_pars(event_size_min, event_size_max, curve_max, curve_min)
        xys = []
        X = np.empty([3,1])
        x = np.arange(0 + height/num_layers,height + height/num_layers, height/num_layers)
        i = 0
        for r, d, s in zip(radii, dirs, signs):
            y1test = y1(x, r, d, s)
        #     print(y1test, x)
            y2test = y2(x, r, d, s)
            if -2.5 < y1test[0] < 2.5 and not any(np.isnan(y1test)):
                X = np.append(X, np.vstack((y1test, np.array([i]*len(y1test)), x )), axis=1)
                i += 1
            if -2.5 < y2test[0] < 2.5 and not any(np.isnan(y2test)):
                X = np.append(X, np.vstack((y2test, np.array([i]*len(y2test)), x )), axis=1)
                i += 1
        X = X[:,1:].T
        np.random.shuffle(X)

        e = np.array([[i,j] for layer in np.arange(num_layers-1) for i in np.argwhere(X[:,2] == layer+1) for j in np.argwhere(X[:,2] == (layer+2)) if (X[i, 0] - np.tan(max_angle/2) < X[j, 0] < X[i, 0] + np.tan(max_angle/2))]).T.squeeze()
        
        # This handles when no edges were constructed. In that case, the randomisation is a do-over
        try:
            y = np.array([int(i[1] == j[1]) for i,j in zip(X[e[0]], X[e[1]])])    
            break
        except:
            pass
        
    print("Current progress: ", np.round((iter/kwargs["num_samples"])*100, 2), "%")
    
    data = Data(x = torch.from_numpy(np.array([X[:,2], X[:,0]]).T).float(), edge_index = torch.from_numpy(e), y = torch.from_numpy(y), pid = torch.from_numpy(X[:,1]))
    return data

def split_dataset(dataset, **kwargs):
    train_dataset, test_dataset = dataset[:int(len(dataset)*kwargs["train_percent"]/100)], dataset[int(len(dataset)*kwargs["train_percent"]/100):]
    return train_dataset, test_dataset

#_________________________________ Visualisations _____________________

def visualise_training_dataset(train_dataset):
    tracks = [len(np.unique(data.pid)) for data in train_dataset]
    fig, axs = plt.subplots(ncols=3)
    fig.set_size_inches(16,4)
    sns.distplot(tracks, ax=axs[0]).set_title("Number of tracks")
    sns.distplot(tracks, ax=axs[1]).set_title("Number of tracks")
    sns.distplot(tracks, ax=axs[2]).set_title("Number of tracks")    