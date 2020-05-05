"""
This file contains some common helper code for the analysis notebooks.
"""

# System
import os
import yaml
import pickle
from collections import namedtuple

# Externals
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset, DataLoader

# Locals
from torch_geometric.data import Batch
from ..datasets.hitgraphs_sparse import HitGraphDataset

#------------------------------------------------------------------------------

def get_output_dir(config):
    return os.path.expandvars(config['output_dir'])

def get_input_dir(config):
    return os.path.expandvars(config['data']['input_dir'])

def load_config_dir(result_dir):
    """Load pickled config saved in a result directory"""
    config_file = os.path.join(result_dir, 'config.pkl')
    with open(config_file, 'rb') as f:
        return pickle.load(f)

def load_id_file(f):
    with np.load(f, allow_pickle=True) as id_data:
        return id_data["I"], id_data["pid"]
    
def load_summaries(config):
    summary_file = os.path.join(get_output_dir(config), 'summaries_0.csv')
    return pd.read_csv(summary_file)

def get_dataset_from_config(config):
    return HitGraphDataset(get_input_dir(config))

def get_dataset_from_path(path):
    return HitGraphDataset(path)

def get_data_loader(config_or_path, n_tasks, task):
    
    # A flexible method to load a dataset from a model config or a simple manual path
    if isinstance(config_or_path, str):
        full_dataset = get_dataset_from_path(config_or_path)
    else:
        full_dataset = get_dataset_from_config(config_or_path)
        
    full_indices = torch.arange(len(full_dataset))
    sub_indices = np.array_split(full_indices,n_tasks)[task]
    sub_dataset = Subset(full_dataset, sub_indices.tolist())
    return DataLoader(sub_dataset, batch_size=1, collate_fn=Batch.from_data_list) #, sub_indices.numpy()

def get_IDs(config, n_tasks, task):
    input_dir = get_input_dir(config)
    all_events = os.listdir(input_dir)
    filenames = sorted([os.path.join(input_dir, f) for f in all_events
                                if f.startswith('event') and f.endswith('_ID.npz')])
    eventnames = sorted([os.path.splitext(f)[0] for f in all_events
                                if f.startswith('event') and not f.endswith('_ID.npz')])
    task_filenames = np.array_split(filenames,n_tasks)[task]
    task_events = np.array_split(eventnames, n_tasks)[task]
    ID_data = [load_id_file(f) for f in task_filenames]
    
    return ID_data, task_events

def get_seed_data_loader(config, n_tasks, task):
    # Take the test set from the back
    full_dataset = get_dataset(config)
    full_indices = torch.arange(len(full_dataset))
    sub_indices = np.array_split(full_indices,n_tasks)[task]
    sub_dataset = Subset(full_dataset, sub_indices.tolist())
    full_filelist = full_dataset.get_filelist()
    return DataLoader(sub_dataset, batch_size=1,
                      collate_fn=Batch.from_data_list), full_filelist