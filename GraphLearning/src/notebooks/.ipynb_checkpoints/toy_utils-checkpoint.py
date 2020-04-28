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
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv
# from torch_geometric.nn import MessagePassing
# import torch_geometric.nn as tnn
# from torch_geometric.utils import add_self_loops, degree
# from torch_scatter import *

# Locals
from torch_geometric.data import Batch
from datasets.hitgraphs import load_graph


# Data Loading

def load_data(train_size=300, test_size=10):
    
    import torch_geometric.data
    
    input_dir = "/global/cscratch1/sd/danieltm/ExaTrkX/node_tracker_data/hitgraphs_med_000/"
    filenames = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                         if f.endswith('.npz') and not f.endswith('_ID.npz')]
    train_graphs = [load_graph(fi) for fi in filenames[:train_size]]
    test_graphs = [load_graph(fi) for fi in filenames[:test_size]]
    train_dataset = [torch_geometric.data.Data(x=torch.from_numpy(di[0]),
                                         edge_index=torch.from_numpy(di[1]), y_edges=torch.from_numpy(di[2]), 
                                         y_params=(torch.from_numpy(di[3][:,0]).unsqueeze(1)), pid=torch.from_numpy(di[4])) for di in train_graphs]
    test_dataset = [torch_geometric.data.Data(x=torch.from_numpy(di[0]),
                                         edge_index=torch.from_numpy(di[1]), y_edges=torch.from_numpy(di[2]), 
                                         y_params=(torch.from_numpy(di[3][:,0]).unsqueeze(1)), pid=torch.from_numpy(di[4])) for di in test_graphs]
    return train_dataset, test_dataset

# Some dumb circle calculations
def y1(x, r, a, sign):
    return sign*np.sqrt(r**2 - a**2) + np.sqrt(r**2 - (x-a)**2)
def y2(x, r, a, sign):
    return sign*np.sqrt(r**2 - a**2) - np.sqrt(r**2 - (x-a)**2)

# Generate random circle / helix parameters
def rand_pars(event_size_min, event_size_max, max_curve, min_curve):
    event_size = int(np.floor(np.random.random(1)*(event_size_max - event_size_min) + event_size_min))
    radii = np.random.random(event_size)*(max_curve - min_curve) + min_curve
    dirs = np.random.random(event_size)*(radii)*2 -radii
    sign_options = np.array([-1,1])
    signs = sign_options[np.rint(np.random.random(event_size)).astype(int)]
    return radii, dirs, signs, event_size


# Generate event data from random parameters
def gen_edge_class(event_size_min, event_size_max, max_curve, min_curve, height, num_layers, max_angle):
    radii, dirs, signs, event_size = rand_pars(event_size_min, event_size_max, max_curve, min_curve)
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
        
    e = np.array([[i,j] for layer in np.arange(num_layers) for i in np.argwhere(X[:,2] == layer) for j in np.argwhere(X[:,2] == (layer+1)) if (X[i, 0] - np.tan(max_angle/2) < X[j, 0] < X[i, 0] + np.tan(max_angle/2))]).T[0]
    y = np.array([int(i[1] == j[1]) for i,j in zip(X[e[0]], X[e[1]])])    
    
    # Normalise
    X = X / 10
    
    data = Data(x = torch.from_numpy(np.array([X[:,2], X[:,0]]).T).float(), edge_index = torch.from_numpy(e), y = torch.from_numpy(y))
    
    return data

# Generate event data from random parameters
def gen_graph_class(event_size_min, event_size_max, max_curve, min_curve, height, num_layers, max_angle):
    while True:
        radii, dirs, signs, event_size = rand_pars(event_size_min, event_size_max, max_curve, min_curve)
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
    
    # Normalise
    X = X / 10
    
    data = Data(x = torch.from_numpy(np.array([X[:,2], X[:,0]]).T).float(), edge_index = torch.from_numpy(e), y = torch.from_numpy(y))
    
    return data

# Generate event data from random parameters
def gen_edge_graph_class(event_size_min, event_size_max, max_curve, min_curve, height, num_layers, max_angle):
    radii, dirs, signs, event_size = rand_pars(event_size_min, event_size_max, max_curve, min_curve)
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
    
    e = np.array([[i,j] for layer in np.arange(num_layers) for i in np.argwhere(X[:,2] == layer) for j in np.argwhere(X[:,2] == (layer+1)) if (X[i, 0] - np.tan(max_angle/2) < X[j, 0] < X[i, 0] + np.tan(max_angle/2))]).T[0]
    y_edge = np.array([int(i[1] == j[1]) for i,j in zip(X[e[0]], X[e[1]])])
    y_graph = np.array(i)
    data = Data(x = torch.from_numpy(np.array([X[:,2], X[:,0]]).T).float(), edge_index = torch.from_numpy(e), y = torch.from_numpy(y_edge), y_graph = torch.from_numpy(y_graph).unsqueeze(-1))
    
    return data

# Generate event data from random parameters
def gen_edge_track_params(event_size_min, event_size_max, max_curve, min_curve, height, num_layers, max_angle):
    radii, dirs, signs, event_size = rand_pars(event_size_min, event_size_max, max_curve, min_curve)
    xys = []
    X = np.empty([6,1])
    x = np.arange(0 + height/num_layers,height + height/num_layers, height/num_layers)
    i = 0
    for r, d, s in zip(radii, dirs, signs):
        y1test = y1(x, r, d, s)
    #     print(y1test, x)
        y2test = y2(x, r, d, s)
        if -2.5 < y1test[0] < 2.5 and not any(np.isnan(y1test)):
            X = np.append(X, np.vstack((y1test, np.array([i]*len(y1test)), x, np.array([r]*len(y1test))/max_curve, np.array([d]*len(y1test))/max_curve, np.array([s]*len(y1test)) )), axis=1)
            i += 1
        if -2.5 < y2test[0] < 2.5 and not any(np.isnan(y2test)):
            X = np.append(X, np.vstack((y2test, np.array([i]*len(y2test)), x, np.array([r]*len(y2test))/max_curve, np.array([d]*len(y2test))/max_curve, np.array([s]*len(y2test)) )), axis=1)
            i += 1
    X = X[:,1:].T
    np.random.shuffle(X)
    
    e = np.array([[i,j] for layer in np.arange(num_layers) for i in np.argwhere(X[:,2] == layer) for j in np.argwhere(X[:,2] == (layer+1)) if (X[i, 0] - np.tan(max_angle/2) < X[j, 0] < X[i, 0] + np.tan(max_angle/2))]).T[0]
    y_edge = np.array([int(i[1] == j[1]) for i,j in zip(X[e[0]], X[e[1]])])
    
    data = Data(x = torch.from_numpy(np.array([X[:,2], X[:,0]]).T).float(), edge_index = torch.from_numpy(e), y = torch.from_numpy(y_edge), y_nodes = torch.from_numpy(np.array([X[:,3], X[:,4], X[:,5]]).T).float())
    
    return data

#________________________________________________________________________

# def plot

#________________________________________________________________________

def make_mlp(input_size, sizes,
             hidden_activation=nn.ReLU,
             output_activation=nn.ReLU,
             layer_norm=False):
    """Construct an MLP with specified fully-connected layers."""
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers
    for i in range(n_layers-1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i+1]))
        layers.append(hidden_activation())
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        layers.append(output_activation())
    return nn.Sequential(*layers)

#__________________________ Vanilla Edge Classifaction Network _____________


class EdgeNetwork(nn.Module):
    """
    A module which computes weights for edges of the graph.
    For each edge, it selects the associated nodes' features
    and applies some fully-connected network layers with a final
    sigmoid activation.
    """
    def __init__(self, input_dim, hidden_dim=8, hidden_activation=nn.Tanh,
                 layer_norm=True):
        super(EdgeNetwork, self).__init__()
        self.network = make_mlp(input_dim*2,
                                [hidden_dim, hidden_dim, hidden_dim, 1],
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)

    def forward(self, x, edge_index):
        # Select the features of the associated nodes
        start, end = edge_index
        x1, x2 = x[start], x[end]
        edge_inputs = torch.cat([x[start], x[end]], dim=1)
        return self.network(edge_inputs).squeeze(-1)

class NodeNetwork(nn.Module):
    """
    A module which computes new node features on the graph.
    For each node, it aggregates the neighbor node features
    (separately on the input and output side), and combines
    them with the node's previous features in a fully-connected
    network to compute the new features.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_activation=nn.Tanh,
                 layer_norm=True):
        super(NodeNetwork, self).__init__()
        self.network = make_mlp(input_dim*3, [hidden_dim, hidden_dim, hidden_dim, output_dim],
                                hidden_activation=hidden_activation,
                                output_activation=hidden_activation,
                                layer_norm=layer_norm)

    def forward(self, x, e, edge_index):
        start, end = edge_index
        # Aggregate edge-weighted incoming/outgoing features
        mi = scatter_add(e[:, None] * x[start], end, dim=0, dim_size=x.shape[0])
        mo = scatter_add(e[:, None] * x[end], start, dim=0, dim_size=x.shape[0])
        node_inputs = torch.cat([mi, mo, x], dim=1)
        return self.network(node_inputs)


class Edge_Class_Net(nn.Module):
    """
    Segment classification graph neural network model.
    Consists of an input network, an edge network, and a node network.
    """
    def __init__(self, input_dim=3, hidden_dim=8, n_graph_iters=3,
                 hidden_activation=nn.Tanh, layer_norm=True):
        super(Edge_Class_Net, self).__init__()
        self.n_graph_iters = n_graph_iters
        # Setup the input network
        self.input_network = make_mlp(input_dim, [hidden_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=layer_norm)
        # Setup the edge network
        self.edge_network = EdgeNetwork(input_dim+hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
        # Setup the node layers
        self.node_network = NodeNetwork(input_dim+hidden_dim, hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)

    def forward(self, inputs):
        """Apply forward pass of the model"""
        # Apply input network to get hidden representation
        x = self.input_network(inputs.x)
        # Shortcut connect the inputs onto the hidden representation
        x = torch.cat([x, inputs.x], dim=-1)
        # Loop over iterations of edge and node networks
        for i in range(self.n_graph_iters):
            # Apply edge network
            e = torch.sigmoid(self.edge_network(x, inputs.edge_index))
            # Apply node network
            x = self.node_network(x, e, inputs.edge_index)
            # Shortcut connect the inputs onto the hidden representation
            x = torch.cat([x, inputs.x], dim=-1)
        # Apply final edge network
        return self.edge_network(x, inputs.edge_index)
    
#_______________ Vanilla Track Counter ________________________
    
class Out_Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Out_Net, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, out_channels)
        self.lin2 = torch.nn.Linear(in_channels, in_channels)

    def forward(self, x, batch):
        x = tnn.global_mean_pool(x, batch)
        x = F.relu(x)
        x = self.lin1(x.float())
        x = F.relu(x)
        
        return x


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.lin = nn.Sequential(torch.nn.Linear(64, 64),  nn.ReLU(), torch.nn.Linear(64, 64),  nn.ReLU(), torch.nn.Linear(64, 64),  nn.ReLU(), torch.nn.Linear(64, 64),  nn.ReLU())
        self.linout = nn.Sequential(torch.nn.Linear(8, 8),  nn.ReLU(), torch.nn.Linear(8, 8),  nn.ReLU(), torch.nn.Linear(8, 8),  nn.ReLU(), torch.nn.Linear(8, 8),  nn.ReLU())
#         self.conv1 = GCNConv(2, 16)
#         self.conv2 = GCNConv(16, 64)
#         self.conv3 = GCNConv(64, 64)
        self.input = nn.Sequential(torch.nn.Linear(2, 64),nn.ReLU())
        self.conv1 = tnn.HypergraphConv(64, 64, use_attention=True)
        self.conv2 = tnn.HypergraphConv(64, 64, use_attention=True)
#         self.conv2 = tnn.nn.Linear(64, 64)
        self.out = Out_Net(64, 12)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.input(x.float())
        x = self.conv1(x.float(), edge_index)
        x = F.relu(x)
        x = self.lin(x.float())
        x = self.conv2(x.float(), edge_index)
        x = F.relu(x)
        x = self.lin(x.float())
        x = self.conv2(x.float(), edge_index)
        x = F.relu(x)
        x = self.lin(x.float())
        x = self.conv2(x.float(), edge_index)
        x = F.relu(x)
        x = self.lin(x.float())
        x = self.conv2(x.float(), edge_index)
        x = F.relu(x)
        x = self.lin(x.float())
        x = self.conv2(x.float(), edge_index)
        x = F.relu(x)
        x = self.out(x, batch)
#         x = F.relu(x)
#         x = self.linout(x.float())
#         return torch.sigmoid(x)
        return x
    

#__________________________ Combined Edge + Counter Classifaction Network _____________


# class EdgeNetwork(nn.Module):
#     """
#     A module which computes weights for edges of the graph.
#     For each edge, it selects the associated nodes' features
#     and applies some fully-connected network layers with a final
#     sigmoid activation.
#     """
#     def __init__(self, input_dim, hidden_dim=8, hidden_activation=nn.Tanh,
#                  layer_norm=True):
#         super(EdgeNetwork, self).__init__()
#         self.network = make_mlp(input_dim*2,
#                                 [hidden_dim, hidden_dim, hidden_dim, 1],
#                                 hidden_activation=hidden_activation,
#                                 output_activation=None,
#                                 layer_norm=layer_norm)

#     def forward(self, x, edge_index):
#         # Select the features of the associated nodes
#         start, end = edge_index
#         x1, x2 = x[start], x[end]
#         edge_inputs = torch.cat([x[start], x[end]], dim=1)
#         return self.network(edge_inputs).squeeze(-1)

# class NodeNetwork(nn.Module):
#     """
#     A module which computes new node features on the graph.
#     For each node, it aggregates the neighbor node features
#     (separately on the input and output side), and combines
#     them with the node's previous features in a fully-connected
#     network to compute the new features.
#     """
#     def __init__(self, input_dim, output_dim, hidden_activation=nn.Tanh,
#                  layer_norm=True):
#         super(NodeNetwork, self).__init__()
#         self.network = make_mlp(input_dim*3, [output_dim]*4,
#                                 hidden_activation=hidden_activation,
#                                 output_activation=hidden_activation,
#                                 layer_norm=layer_norm)

#     def forward(self, x, e, edge_index):
#         start, end = edge_index
#         # Aggregate edge-weighted incoming/outgoing features
#         mi = scatter_add(e[:, None] * x[start], end, dim=0, dim_size=x.shape[0])
#         mo = scatter_add(e[:, None] * x[end], start, dim=0, dim_size=x.shape[0])
#         node_inputs = torch.cat([mi, mo, x], dim=1)
#         return self.network(node_inputs)

# class Out_Net(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Out_Net, self).__init__()
#         self.lin1 = torch.nn.Linear(in_channels, out_channels)
#         self.lin2 = torch.nn.Linear(in_channels, in_channels)

#     def forward(self, x, batch):
#         x = tnn.global_mean_pool(x, batch)
#         x = self.lin2(x.float())
#         x = F.relu(x)
#         x = self.lin1(x.float())
        
#         return x

class Edge_Graph_Class_Net(nn.Module):
    """
    Segment classification graph neural network model.
    Consists of an input network, an edge network, and a node network.
    """
    def __init__(self, input_dim=3, hidden_dim=8, n_graph_iters=3,
                 output_dim = 8, hidden_activation=nn.Tanh, layer_norm=True):
        super(Edge_Graph_Class_Net, self).__init__()
        self.n_graph_iters = n_graph_iters
        # Setup the input network
        self.input_network = make_mlp(input_dim, [hidden_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=layer_norm)
        # Setup the edge network
        self.edge_network = EdgeNetwork(input_dim+hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
        # Setup the node layers
        self.node_network = NodeNetwork(input_dim+hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
        self.out_network = Out_Net(input_dim+hidden_dim, output_dim)

    def forward(self, inputs):
        """Apply forward pass of the model"""
        # Apply input network to get hidden representation
        x = self.input_network(inputs.x)
        # Shortcut connect the inputs onto the hidden representation
        x = torch.cat([x, inputs.x], dim=-1)
        # Loop over iterations of edge and node networks
        for i in range(self.n_graph_iters):
            # Apply edge network
            e = torch.sigmoid(self.edge_network(x, inputs.edge_index))
            # Apply node network
            x = self.node_network(x, e, inputs.edge_index)
            # Shortcut connect the inputs onto the hidden representation
            x = torch.cat([x, inputs.x], dim=-1)
        # Apply final edge network
        o = self.out_network(x, inputs.batch)
        return self.edge_network(x, inputs.edge_index), o
    
    
#__________________ Combined Edge & Track Param Classifier ___________


class Edge_Track_Net(nn.Module):
    """
    Segment classification graph neural network model.
    Consists of an input network, an edge network, and a node network.
    """
    def __init__(self, input_dim=3, hidden_dim=8, n_graph_iters=3,
                 output_dim=3, hidden_activation=nn.Tanh, layer_norm=True):
        super(Edge_Track_Net, self).__init__()
        self.n_graph_iters = n_graph_iters
        # Setup the input network
        self.input_network = make_mlp(input_dim, [hidden_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=layer_norm)
        # Setup the edge network
        self.edge_network = EdgeNetwork(input_dim+hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
        # Setup the node layers
        self.node_network = NodeNetwork(input_dim+hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=False)
        
        self.output_network = make_mlp(input_dim+hidden_dim, [hidden_dim, output_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=False)

    def forward(self, inputs):
        """Apply forward pass of the model"""
        # Apply input network to get hidden representation
        x = self.input_network(inputs.x)
        # Shortcut connect the inputs onto the hidden representation
        x = torch.cat([x, inputs.x], dim=-1)
        # Loop over iterations of edge and node networks
        for i in range(self.n_graph_iters):
            # Apply edge network
            e = torch.sigmoid(self.edge_network(x, inputs.edge_index))
            # Apply node network
            x = self.node_network(x, e, inputs.edge_index)
            # Shortcut connect the inputs onto the hidden representation
            x = torch.cat([x, inputs.x], dim=-1)
        # Apply final edge network
        return self.edge_network(x, inputs.edge_index), self.output_network(x)
    


#___________________________________________________________________
    
# class GCNConv(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
#         self.lin = torch.nn.Linear(in_channels, out_channels)

#     def forward(self, x, edge_index):
#         # x has shape [N, in_channels]
#         # edge_index has shape [2, E]

#         # Step 1: Add self-loops to the adjacency matrix.
#         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

#         # Step 2: Linearly transform node feature matrix.
#         x = self.lin(x.float())

#         # Step 3-5: Start propagating messages.
#         return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

#     def message(self, x_j, edge_index, size):
#         # x_j has shape [E, out_channels]

#         # Step 3: Normalize node features.
#         row, col = edge_index
#         deg = degree(row, size[0], dtype=x_j.dtype)
#         deg_inv_sqrt = deg.pow(-0.5)
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

#         return norm.view(-1, 1) * x_j

#     def update(self, aggr_out):
#         # aggr_out has shape [N, out_channels]

#         # Step 5: Return new node embeddings.
#         return aggr_out

# class Out_Net(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Out_Net, self).__init__()
#         self.lin = torch.nn.Linear(in_channels, out_channels)

#     def forward(self, x, batch):
#         x = scatter_mean(x, batch, dim=0)
#         x = self.lin(x.float())
        


# class Net(torch.nn.Module):
#     def __init__(self, dataset):
#         super(Net, self).__init__()
#         self.conv1 = GCNConv(2, 16)
#         self.out = Out_Net(16, 2)

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch

#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.out(x, batch)
#         print(x)
# #         return F.log_softmax(x, dim=1)
#         return F.sigmoid(x)
    
