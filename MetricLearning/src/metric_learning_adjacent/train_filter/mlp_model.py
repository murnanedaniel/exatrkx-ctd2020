import torch
import torch.nn as nn

import time
import numpy as np


#########################################
#               CORE MODELS             #
#########################################

class Simple(nn.Module):
    def __init__(self, nb_hidden, nb_layer, input_dim, emb_dim=3):
        super(Simple, self).__init__()
        self.fc = nn.Linear(input_dim, emb_dim)

    def forward(self, hits):
        return self.fc(hits)

class Edge_MLP(nn.Module):
    def __init__(self, nb_hidden, nb_layer, input_dim, emb_dim):
        super(Edge_MLP, self).__init__()

        self.norm_set = False

        input_dim = input_dim+4 # add 4 for augmented features
        self.input_dim = input_dim
        self.emb_dim = emb_dim

        layers = [nn.Linear(input_dim, nb_hidden)]
        ln = [nn.Linear(nb_hidden, nb_hidden) for _ in range(nb_layer-1)]
        layers.extend(ln)
        self.layers = nn.ModuleList(layers)
        self.act1 = nn.ReLU()

        self.final_layer = nn.Linear(nb_hidden, 1)
        self.act2 = nn.Sigmoid()
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, hits):
        hits = augment_features(hits)
        hits = self.normalize(hits)
        for l in self.layers:
            hits = self.act1(l(hits))
            # hits = self.dropout(hits)
        hits = self.final_layer(hits)
        hits = self.act2(hits).squeeze()
        return hits

    def normalize(self, hits):
        try:
            hits = (hits-self.mean) / (self.std + 10**-9)
        except:
            self.mean = self.mean.to(device=hits.device)
            self.std  = self.std.to(device=hits.device)
            hits = (hits-self.mean) / (self.std + 10**-9)
        return hits
            
    def set_norm(self, mean, std):
        self.norm_set = True
        self.mean = mean
        self.std  = std

#####################################################
#                   FEATURE AUGMENT                 #
#####################################################
def augment_features(hit_pairs):
    '''
    Augment hits with features derived from TrackML physics
    '''
    nb_feats_one_hit = hit_pairs.size(1) // 2
    x1, y1, z1 = get_xyz(hit_pairs[:, :nb_feats_one_hit])
    x2, y2, z2 = get_xyz(hit_pairs[:, nb_feats_one_hit:])

    dr   = compute_dr(x1, y1, x2, y2)
    dphi = compute_dphi(x1, y1, x2, y2)
    rho  = compute_rho(dr, dphi)

    z0 = compute_z0(x1, y1, z1, x2, y2, z2)

    aug_feats = torch.stack((dr, dphi, rho, z0), dim=1)
    hit_pairs = torch.cat((hit_pairs, aug_feats), dim=1)
    return hit_pairs

def compute_dr(x1, y1, x2, y2):
    dr = torch.pow(torch.pow(x2-x1, 2) + torch.pow(y2-y1, 2), 0.5)
    return dr

def compute_dphi(x1, y1, x2, y2):
    dphi = torch.acos(torch.cos(torch.atan2(y2, x2)-torch.atan2(y1,x1)))
    return dphi

def compute_rho(dr, dphi):
    rho = 0.5 * dr / (torch.sin(dphi) + 10**-8)
    return rho

def compute_z0(x1, y1, z1, x2, y2, z2):
    r1 = compute_r(x1, y1)
    r2 = compute_r(x2, y2)

    dr = r2 - r1
    dz = z2 - z1
    z0 = z1 - r1 * (dz / (dr + 10**-8))
    return z0

def compute_r(x, y):
    return torch.pow(torch.pow(x, 2) + torch.pow(y, 2), 0.5)

def get_xyz(hits):
    x = hits[:,0]/1000
    y = hits[:,1]/1000
    z = hits[:,2]/1000
    return x, y, z
