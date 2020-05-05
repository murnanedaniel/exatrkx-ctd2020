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

class MLP(nn.Module):
    def __init__(self,
                 nb_hidden,
                 nb_layer,
                 input_dim,
                 mean,
                 std,
                 emb_dim=3):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, nb_hidden)]
        ln = [nn.Linear(nb_hidden, nb_hidden) for _ in range(nb_layer-1)]
        layers.extend(ln)
        self.layers = nn.ModuleList(layers)
        self.emb_layer = nn.Linear(nb_hidden, emb_dim)
        self.act = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.7)
        self.mean = torch.FloatTensor(mean).to(torch.float)
        self.std = torch.FloatTensor(std).to(torch.float)

    def forward(self, hits):
        hits = self.normalize(hits)
        for l in self.layers:
            hits = l(hits)
            hits = self.act(hits)
            # hits = self.dropout(hits)
        hits = self.emb_layer(hits)
        return hits

    def normalize(self, hits):
        try:
            hits = (hits-self.mean) / (self.std + 10**-9)
        except:
            self.mean = self.mean.to(device=hits.device)
            self.std  = self.std.to(device=hits.device)
            hits = (hits-self.mean) / (self.std + 10**-9)
        return hits
