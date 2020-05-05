import math
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset

#####################################
#               DATASET             #
#####################################

class Hit_Pair_Dataset(Dataset):
    def __init__(self, data_filepath, nb_samples):
        with open(data_filepath, 'rb') as f:
            dataset = pickle.load(f)

        try:
            self.hits_a = np.array(dataset['hits_a'][:nb_samples],
                                   dtype=np.float32)
            self.hits_b = np.array(dataset['hits_b'][:nb_samples],
                                   dtype=np.float32)
            self.target = np.array(dataset['target'][:nb_samples],
                                   dtype=np.float32)
        except:
            dim = (dataset.shape[1]-1)//2
            self.hits_a = dataset[:nb_samples,:dim]
            self.hits_b = dataset[:nb_samples,dim:2*dim]
            self.target = dataset[:nb_samples,-1]

    def __getitem__(self, index):
        h_a = self.hits_a[index]
        h_b = self.hits_b[index]
        h = np.concatenate((h_a, h_b), axis=0)
        t   = self.target[index]
        return h, t

    def __len__(self):
        return len(self.hits_a)

    def get_input_dim(self):
        h, t = self[0]
        input_dim = len(h)
        return input_dim
