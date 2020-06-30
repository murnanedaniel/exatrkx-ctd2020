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

# Locals
from GraphLearning.src.models import get_model
from GraphLearning.src.datasets.hitgraphs_sparse import HitGraphDataset

def get_output_dirs(config):
    data_dir = os.path.expandvars(config['data_storage_path'])
    artifact_dir = os.path.join(data_dir, "artifacts", config['name'])
    doublet_artifacts = os.path.join(artifact_dir, "doublet_gnn")
    triplet_artifacts = os.path.join(artifact_dir, "triplet_gnn")
    return doublet_artifacts, triplet_artifacts

def get_input_dir(config):
    return os.path.expandvars(config['data']['input_dir'])

def load_config_file(config_file):
    """Load configuration from a specified yaml config file path"""
    with open(config_file) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_config_dir(result_dir):
    """Load pickled config saved in a result directory"""
    config_file = os.path.join(result_dir, 'config.pkl')
    with open(config_file, 'rb') as f:
        return pickle.load(f)

# Back-compat
load_config = load_config_file

def load_summaries(config):
    doublet_artifacts, triplet_artifacts = get_output_dirs(config)
    doublet_summary_file = os.path.join(doublet_artifacts, 'summaries_0.csv')
    triplet_summary_file = os.path.join(triplet_artifacts, 'summaries_0.csv')
    return pd.read_csv(doublet_summary_file), pd.read_csv(triplet_summary_file)

def load_model(config, reload_epoch):
    """deprecated"""
    model_config = config['model']
    model_config.pop('loss_func', None)
    model = get_model(**model_config)

    # Reload specified model checkpoint
    output_dir = get_output_dir(config)
    checkpoint_file = os.path.join(output_dir, 'checkpoints',
                                   'model_checkpoint_%03i.pth.tar' % reload_epoch)
    model.load_state_dict(torch.load(checkpoint_file, map_location='cpu')['model'])
    return model

def get_dataset(config):
    return HitGraphDataset(get_input_dir(config))

def get_test_data_loader(config, n_test=16, batch_size=1):
    # Take the test set from the back
    full_dataset = get_dataset(config)
    test_indices = len(full_dataset) - 1 - torch.arange(n_test)
    test_dataset = Subset(full_dataset, test_indices.tolist())
    full_filelist = full_dataset.get_filelist()
    return DataLoader(test_dataset, batch_size=batch_size,
                      collate_fn=Batch.from_data_list), full_filelist

def load_triplets(test_loader, filelist):
    graph_dataset = test_loader.dataset
    graph_indices = np.array([g.i for g in graph_dataset])
    filelist = np.array(filelist)
    graph_names = filelist[graph_indices]
    
    return graph_dataset, graph_names
    
def save_triplets(test_loader, test_preds, filelist, threshold=0.7, output_dir = "/global/u2/d/danieltm/ExaTrkX/exatrkx-work/gnn_pytorch/notebooks/XY_Triplets/"):
    
    graph_dataset, graph_names = load_triplets(test_loader, filelist)
    
    
    for graph_data, graph_name, test_pred in zip(graph_dataset, graph_names, test_preds):
        g_ID = np.load(graph_name[:-4] + "_ID.npz", allow_pickle=True)["I"]
        e, o = graph_data.edge_index.numpy(), test_pred.numpy()
        triplet_IDs = np.hstack([g_ID[:,e[0,:]].T, g_ID[:,e[1,:]].T])[:,[0,1,3]]
        triplet_preds = triplet_IDs[o > threshold]

        triplet_list = np.c_[np.array([os.path.basename(graph_name)[5:-6]]*len(triplet_preds), dtype=np.int64), triplet_preds.astype(np.int64)]
        filename = output_dir + os.path.basename(graph_name)[:-6]
        with open(filename,'ab') as f:    
            np.savetxt(f, triplet_list, fmt='%s', delimiter=",")

@torch.no_grad()
def apply_model(model, data_loader):
    preds, targets = [], []
    for batch in data_loader:
        preds.append(torch.sigmoid(model(batch)).squeeze(0))
        targets.append(batch.y.squeeze(0))
    return preds, targets

@torch.no_grad()
def apply_dense_model(model, data_loader):
    preds, targets = [], []
    for inputs, target in data_loader:
        preds.append(model(inputs).squeeze(0))
        targets.append(target.squeeze(0))
    return preds, targets

# Define our Metrics class as a namedtuple
Metrics = namedtuple('Metrics', ['accuracy', 'precision', 'recall', 'f1',
                                 'prc_precision', 'prc_recall', 'prc_thresh',
                                 'roc_fpr', 'roc_tpr', 'roc_thresh', 'roc_auc'])

def compute_metrics(preds, targets, threshold=0.5):
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    # Decision boundary metrics
    y_pred, y_true = (preds > threshold), (targets > threshold)
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    precision, recall, f1, support = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')
    #precision = sklearn.metrics.precision_score(y_true, y_pred)
    #recall = sklearn.metrics.recall_score(y_true, y_pred)
    # Precision recall curves
    prc_precision, prc_recall, prc_thresh = sklearn.metrics.precision_recall_curve(y_true, preds)
    # ROC curve
    roc_fpr, roc_tpr, roc_thresh = sklearn.metrics.roc_curve(y_true, preds)
    roc_auc = sklearn.metrics.auc(roc_fpr, roc_tpr)
    # Organize metrics into a namedtuple
    return Metrics(accuracy=accuracy, precision=precision, recall=recall, f1=f1,
                   prc_precision=prc_precision, prc_recall=prc_recall, prc_thresh=prc_thresh,
                   roc_fpr=roc_fpr, roc_tpr=roc_tpr, roc_thresh=roc_thresh, roc_auc=roc_auc)

def plot_train_history(summaries, figsize=(12, 10), loss_yscale='linear'):
    axs, fig = create_train_history(summaries, figsize, loss_yscale)

    plt.tight_layout()

def save_train_history(summaries, output_dir, name, format="pdf", figsize=(12, 10), loss_yscale='linear'):
    
    axs, fig = create_train_history(summaries, figsize, loss_yscale)
    
    output_name = os.path.join(output_dir, name+'_train_history.'+format)
    
    fig.savefig(output_name, format=format)

def create_train_history(summaries, figsize=(12, 10), loss_yscale='linear'):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    axs = axs.flatten()

    # Plot losses
    axs[0].plot(summaries.epoch, summaries.train_loss, label='Train')
    axs[0].plot(summaries.epoch, summaries.valid_loss, label='Validation')
    axs[0].set_yscale(loss_yscale)
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend(loc=0)

    # Plot accuracies
    axs[1].plot(summaries.epoch, summaries.valid_acc, label='Validation')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_ylim(bottom=0, top=1)
    axs[1].legend(loc=0)

    # Plot model weight norm
    axs[2].plot(summaries.epoch, summaries.l2)
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Model L2 weight norm')

    # Plot learning rate
    axs[3].plot(summaries.epoch, summaries.lr)
    axs[3].set_xlabel('Epoch')
    axs[3].set_ylabel('Learning rate')

    return axs, fig

def plot_metrics(preds, targets, metrics, roc_scale="linear", tf_scale='linear')
    
    axs, fig = create_metrics(preds, targets, metrics, roc_scale, tf_scale)
    plt.tight_layout()
    
def save_metrics(summaries, output_dir, name, preds, targets, metrics, format="pdf", roc_scale="linear", tf_scale='linear')
    
    axs, fig = create_metrics(preds, targets, metrics, roc_scale, tf_scale)
    output_name = os.path.join(output_dir, name+'_metrics.'+format)
    fig.savefig(output_name, format=format)

def create_metrics(preds, targets, metrics, roc_scale="linear", tf_scale='linear'):
    # Prepare the values
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    labels = targets > 0.5

    # Create the figure
    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(16,5))

    # Plot the model outputs
    binning=dict(bins=25, range=(0,1), histtype='step', log=True)
    ax0.hist(preds[labels==False], label='fake', **binning)
    ax0.hist(preds[labels==True], label='real', **binning)
    ax0.set_xlabel('Model output')
    ax0.legend(loc=0)

    # Plot precision and recall
    ax1.plot(metrics.prc_thresh, metrics.prc_precision[:-1], label='purity')
    ax1.plot(metrics.prc_thresh, metrics.prc_recall[:-1], label='efficiency')
    ax1.set_xlabel('Model threshold')
    ax1.set_xscale(tf_scale)
    ax1.legend(loc=0)

    # Plot the ROC curve
    ax2.plot(metrics.roc_fpr, metrics.roc_tpr)
    ax2.plot([0, 1], [0, 1], '--')
    ax2.set_xlabel('False positive rate')
    ax2.set_ylabel('True positive rate')
    ax2.set_title('ROC curve, AUC = %.3f' % metrics.roc_auc)
    ax2.set_xscale(roc_scale)

    return (ax0, ax1, ax2), fig
    

def plot_outputs_roc(preds, targets, metrics):
    # Prepare the values
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    labels = targets > 0.5

    # Create the figure
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12,5))

    # Plot the model outputs
    binning=dict(bins=25, range=(0,1), histtype='step', log=True)
    ax0.hist(preds[labels==False], label='fake', **binning)
    ax0.hist(preds[labels==True], label='real', **binning)
    ax0.set_xlabel('Model output')
    ax0.legend(loc=0)

    # Plot the ROC curve
    ax1.plot(metrics.roc_fpr, metrics.roc_tpr)
    ax1.plot([0, 1], [0, 1], '--')
    ax1.set_xlabel('False positive rate')
    ax1.set_ylabel('True positive rate')
    ax1.set_title('ROC curve, AUC = %.3f' % metrics.roc_auc)
    plt.tight_layout()

def draw_sample(X, Ri, Ro, y, cmap='bwr_r', alpha_labels=True, figsize=(15, 7)):
    # Select the i/o node features for each segment
    feats_o = X[np.where(Ri.T)[1]]
    feats_i = X[np.where(Ro.T)[1]]

    # Prepare the figure
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)
    cmap = plt.get_cmap(cmap)

    # Draw the hits (r, phi, z)
    ax0.scatter(X[:,2], X[:,0], c='k')
    ax1.scatter(X[:,1], X[:,0], c='k')

    # Draw the segments
    for j in range(y.shape[0]):
        if alpha_labels:
            seg_args = dict(c='k', alpha=float(y[j]))
        else:
            seg_args = dict(c=cmap(float(y[j])))
        ax0.plot([feats_o[j,2], feats_i[j,2]],
                 [feats_o[j,0], feats_i[j,0]], '-', **seg_args)
        ax1.plot([feats_o[j,1], feats_i[j,1]],
                 [feats_o[j,0], feats_i[j,0]], '-', **seg_args)
    # Adjust axes
    ax0.set_xlabel('$z$')
    ax1.set_xlabel('$\phi$')
    ax0.set_ylabel('$r$')
    ax1.set_ylabel('$r$')
    plt.tight_layout()

def draw_sample_xy(hits, edges, preds, labels, cut=0.5, figsize=(16, 16)):
    x = hits[:,0] * np.cos(hits[:,1])
    y = hits[:,0] * np.sin(hits[:,1])
    fig, ax0 = plt.subplots(figsize=figsize)

    # Draw the hits
    ax0.scatter(x, y, s=2, c='k')

    # Draw the segments
    for j in range(labels.shape[0]):

        # False negatives
        if preds[j] < cut and labels[j] > cut:
            ax0.plot([x[edges[0,j]], x[edges[1,j]]],
                     [y[edges[0,j]], y[edges[1,j]]],
                     '--', c='b')

        # False positives
        if preds[j] > cut and labels[j] < cut:
            ax0.plot([x[edges[0,j]], x[edges[1,j]]],
                     [y[edges[0,j]], y[edges[1,j]]],
                     '-', c='r', alpha=preds[j])

        # True positives
        if preds[j] > cut and labels[j] > cut:
            ax0.plot([x[edges[0,j]], x[edges[1,j]]],
                     [y[edges[0,j]], y[edges[1,j]]],
                     '-', c='k', alpha=preds[j])

    return fig, ax0

def draw_triplets_xy(hits, edges, preds, labels, cut=0.5, figsize=(16, 16)):
    xi, yi = [hits[:,0] * np.cos(hits[:,1]), hits[:,0] * np.sin(hits[:,1])]
    xo, yo = [hits[:,3] * np.cos(hits[:,4]), hits[:,3] * np.sin(hits[:,4])]
    fig, ax0 = plt.subplots(figsize=figsize)

    #Draw the hits
    ax0.scatter(xi, yi, s=2, c='k')

    # Draw the segments
    for j in range(labels.shape[0]):

        # False negatives
        if preds[j] < cut and labels[j] > cut:
            ax0.plot([xi[edges[0,j]], xo[edges[0,j]]],
                     [yi[edges[0,j]], yo[edges[0,j]]],
                     '--', c='b')

        # False positives
        if preds[j] > cut and labels[j] < cut:
            ax0.plot([xi[edges[0,j]], xo[edges[0,j]]],
                     [yi[edges[0,j]], yo[edges[0,j]]],
                     '-', c='r', alpha=preds[j])

        # True positives
        if preds[j] > cut and labels[j] > cut:
            ax0.plot([xi[edges[0,j]], xo[edges[0,j]]],
                     [yi[edges[0,j]], yo[edges[0,j]]],
                     '-', c='k', alpha=preds[j])

    return fig, ax0

def draw_triplets_xy_antiscore(hits, edges, preds, labels, doublet=None, cut=0.5, figsize=(16, 16)):
    xi, yi = [hits[:,0] * np.cos(hits[:,1]), hits[:,0] * np.sin(hits[:,1])]
    xo, yo = [hits[:,3] * np.cos(hits[:,4]), hits[:,3] * np.sin(hits[:,4])]
    scores = hits[:,6]
    fig, ax0 = plt.subplots(figsize=figsize)

#     Draw the hits
    ax0.scatter(xi, yi, s=2, c='k')

    # Draw the segments
    #for j in range(labels.shape[0]):
    if doublet is None:
        for j in range(len(labels)):

            # False negatives
            if preds[j] < cut and labels[j] > cut:
                ax0.plot([xi[edges[0,j]], xo[edges[0,j]]],
                         [yi[edges[0,j]], yo[edges[0,j]]],
                         '--', c='b', alpha=(1-scores[edges[0,j]]))
                ax0.plot([xi[edges[1,j]], xo[edges[1,j]]],
                         [yi[edges[1,j]], yo[edges[1,j]]],
                         '--', c='b', alpha=(1-scores[edges[1,j]]))

            # False positives
            if preds[j] > cut and labels[j] < cut:
                ax0.plot([xi[edges[0,j]], xo[edges[0,j]]],
                         [yi[edges[0,j]], yo[edges[0,j]]],
                         '-', c='r', alpha=preds[j])
            if preds[j] > cut and labels[j] < cut:
                ax0.plot([xi[edges[1,j]], xo[edges[1,j]]],
                         [yi[edges[1,j]], yo[edges[1,j]]],
                         '-', c='r', alpha=preds[j])

            # True positives
            if preds[j] > cut and labels[j] > cut:
                ax0.plot([xi[edges[0,j]], xo[edges[0,j]]],
                         [yi[edges[0,j]], yo[edges[0,j]]],
                         '-', c='k', alpha=(1-scores[edges[0,j]]))
            if preds[j] > cut and labels[j] > cut:
                ax0.plot([xi[edges[1,j]], xo[edges[1,j]]],
                         [yi[edges[1,j]], yo[edges[1,j]]],
                         '-', c='k', alpha=(1-scores[edges[1,j]]))

    return fig, ax0

def draw_triplets_xy_antiscore_cut_edges(hits, edges, preds, labels, doublet=None, cut=0.5, figsize=(16, 16)):
    xi, yi = [hits[:,0] * np.cos(hits[:,1]), hits[:,0] * np.sin(hits[:,1])]
    xo, yo = [hits[:,3] * np.cos(hits[:,4]), hits[:,3] * np.sin(hits[:,4])]
    scores = hits[:,6]
    fig, ax0 = plt.subplots(figsize=figsize)

#     Draw the hits
    ax0.scatter(xi, yi, s=2, c='k')

    # Draw the segments
    #for j in range(labels.shape[0]):
    if doublet is None:
        for j in range(len(labels)):

            # False negatives that doublet identifyer would have correctly identified
            if preds[j] < cut and labels[j] > cut and scores[edges[0,j]] > cut:
                ax0.plot([xi[edges[0,j]], xo[edges[0,j]]],
                         [yi[edges[0,j]], yo[edges[0,j]]],
                         '--', c='r')
            if preds[j] < cut and labels[j] > cut and scores[edges[1,j]] > cut:
                ax0.plot([xi[edges[1,j]], xo[edges[1,j]]],
                         [yi[edges[1,j]], yo[edges[1,j]]],
                         '--', c='r')
                
            # True negatives that doublet identifyer would have incorrectly identified
            if preds[j] < cut and labels[j] < cut and scores[edges[0,j]] > cut:
                ax0.plot([xi[edges[0,j]], xo[edges[0,j]]],
                         [yi[edges[0,j]], yo[edges[0,j]]],
                         '--', c='k')
            if preds[j] < cut and labels[j] < cut and scores[edges[1,j]] > cut:
                ax0.plot([xi[edges[1,j]], xo[edges[1,j]]],
                         [yi[edges[1,j]], yo[edges[1,j]]],
                         '--', c='k')

            # False positives that doublet identifyer would have correctly identified
            if preds[j] > cut and labels[j] < cut and scores[edges[0,j]] < cut:
                ax0.plot([xi[edges[0,j]], xo[edges[0,j]]],
                         [yi[edges[0,j]], yo[edges[0,j]]],
                         '-', c='r', alpha=preds[j])
            if preds[j] > cut and labels[j] < cut and scores[edges[1,j]] < cut:
                ax0.plot([xi[edges[1,j]], xo[edges[1,j]]],
                         [yi[edges[1,j]], yo[edges[1,j]]],
                         '-', c='r', alpha=preds[j])

            # True positives that doublet identifyer would have incorrectly identified
            if preds[j] > cut and labels[j] > cut and scores[edges[0,j]] < cut:
                ax0.plot([xi[edges[0,j]], xo[edges[0,j]]],
                         [yi[edges[0,j]], yo[edges[0,j]]],
                         '-', c='k')
            if preds[j] > cut and labels[j] > cut and scores[edges[1,j]] < cut:
                ax0.plot([xi[edges[1,j]], xo[edges[1,j]]],
                         [yi[edges[1,j]], yo[edges[1,j]]],
                         '-', c='k')

    return fig, ax0

def draw_triplets_xy_antiscore_cut(hits, edges, preds, labels, cut=0.5, figsize=(16, 16)):
    xi, yi = hits[:,0] * np.cos(hits[:,1]), hits[:,0] * np.sin(hits[:,1])
    xo, yo = hits[:,3] * np.cos(hits[:,4]), hits[:,3] * np.sin(hits[:,4])
    scores = hits[:,6]
    fig, ax0 = plt.subplots(figsize=figsize)
    
    tf_mul = tf_multiplicity(hits, edges, preds, labels, cut)
#     Draw the hits
    ax0.scatter(xi, yi, s=2, c='k')
    ax0.scatter(xo, yo, s=2, c='k')
    
    w = 0
    l = 0
    # Draw the segments
    for j in range(len(xi)):
        # Gold standards that the doublet model would have incorrectly identified
        if (tf_mul[j][0]>0 and tf_mul[j][1]==0 
            and tf_mul[j][3]==0 and scores[j] < cut):
            ax0.plot([xi[j],xo[j]],[yi[j],yo[j]],
                     '-', c='k')
            w += 1
        # Gold standards with missed tracks that the doublet model would have incorrectly identified
        if (tf_mul[j][0]>0 and tf_mul[j][1]==0 
            and tf_mul[j][3]>0 and scores[j] < cut):
            ax0.plot([xi[j],xo[j]],[yi[j],yo[j]],
                     '--', c='k')
            w += 1
            
        # Silver standard positives that the doublet model would have incorrectly identified
        if (tf_mul[j][0]>0 and tf_mul[j][1]>0 and scores[j] < cut):
            ax0.plot([xi[j],xo[j]],[yi[j],yo[j]],
                     '-.', c='k')
            w += 1
        
        # Totally missed edges that the doublet model would have correctly identified
        if (tf_mul[j][0]==0 and tf_mul[j][1]==0 and 
            tf_mul[j][3]>0 and scores[j] > cut):
            ax0.plot([xi[j],xo[j]],[yi[j],yo[j]],
                     '--', c='r')
            l += 1
            
        # Totally incorrect edges that the doublet model would have correctly identified
        if (tf_mul[j][0]==0 and tf_mul[j][1]>0 and
           scores[j] < cut):
            ax0.plot([xi[j],xo[j]],[yi[j],yo[j]],
                     '-', c='r')
            l += 1
        
        # True negatives that the doublet model would have incorrectly identified
        if (tf_mul[j][0]==0 and tf_mul[j][1]==0 and 
           tf_mul[j][2] > 0 and tf_mul[j][3]==0 and scores[j] > cut):
            ax0.plot([xi[j],xo[j]],[yi[j],yo[j]],
                     ':', c='k')
            w += 1
    print("Overperforms by: ", w, ", underperforms by: ", l, ".")
    return fig, ax0

def draw_triplets_xy_score(hits, edges, preds, labels, doublet=None, cut=0.5, figsize=(16, 16)):
    xi, yi = [hits[:,0] * np.cos(hits[:,1]), hits[:,0] * np.sin(hits[:,1])]
    xo, yo = [hits[:,3] * np.cos(hits[:,4]), hits[:,3] * np.sin(hits[:,4])]
    scores = hits[:,6]
    fig, ax0 = plt.subplots(figsize=figsize)

#     Draw the hits
    ax0.scatter(xi, yi, s=2, c='k')

    # Draw the segments
    #for j in range(labels.shape[0]):
    if doublet is None:
        for j in range(len(labels)):

            # False negatives
            if preds[j] < cut and labels[j] > cut:
                ax0.plot([xi[edges[0,j]], xo[edges[0,j]]],
                         [yi[edges[0,j]], yo[edges[0,j]]],
                         '--', c='b', alpha=(scores[edges[0,j]]))
                ax0.plot([xi[edges[1,j]], xo[edges[1,j]]],
                         [yi[edges[1,j]], yo[edges[1,j]]],
                         '--', c='b', alpha=(scores[edges[1,j]]))

            # False positives
            if preds[j] > cut and labels[j] < cut:
                ax0.plot([xi[edges[0,j]], xo[edges[0,j]]],
                         [yi[edges[0,j]], yo[edges[0,j]]],
                         '-', c='r', alpha=preds[j])
            if preds[j] > cut and labels[j] < cut:
                ax0.plot([xi[edges[1,j]], xo[edges[1,j]]],
                         [yi[edges[1,j]], yo[edges[1,j]]],
                         '-', c='r', alpha=preds[j])

            # True positives
            if preds[j] > cut and labels[j] > cut:
                ax0.plot([xi[edges[0,j]], xo[edges[0,j]]],
                         [yi[edges[0,j]], yo[edges[0,j]]],
                         '-', c='k', alpha=(scores[edges[0,j]]))
            if preds[j] > cut and labels[j] > cut:
                ax0.plot([xi[edges[1,j]], xo[edges[1,j]]],
                         [yi[edges[1,j]], yo[edges[1,j]]],
                         '-', c='k', alpha=(scores[edges[1,j]]))

    return fig, ax0

def add_multiplicity(hits, edges, preds, cut=0.5):
    mul = np.zeros(len(hits))
    w_mul = np.zeros(len(hits))
    for edge, pred in zip(edges.T, preds) :
        if pred > cut: 
            mul[edge[0]] += 1
            mul[edge[1]] += 1
            w_mul[edge[0]] += pred**2
            w_mul[edge[1]] += pred**2
#     return np.concatenate([hits, np.array([mul]).T], axis=1)
    return mul, w_mul

def draw_triplets_mul_xy(hits, edges, preds, labels, cut=0.5, figsize=(16, 16)):
    xi, yi = hits[:,0] * np.cos(hits[:,1]), hits[:,0] * np.sin(hits[:,1])
    xo, yo = hits[:,3] * np.cos(hits[:,4]), hits[:,3] * np.sin(hits[:,4])
    scores = hits[:,6]
    fig, ax0 = plt.subplots(figsize=figsize)
    cmap = plt.cm.seismic
    mul, w_mul = add_multiplicity(hits, edges, preds, cut)
#     Draw the hits
    ax0.scatter(xi, yi, s=2, c='k')
    ax0.scatter(xo, yo, s=2, c='k')

    # Draw the segments
    #for j in range(labels.shape[0]):
    for j in range(len(xi)):
        if (mul[j] == 1 or mul[j] == 2):
#             if (w_mul[j]/mul[j]) > cut:
            ax0.plot([xi[j],xo[j]],[yi[j],yo[j]], color=cmap(1-scores[j]))


    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])  # only needed for matplotlib < 3.1
    fig.colorbar(sm)
    return fig, ax0

def tf_multiplicity(hits, edges, preds, labels, cut=0.5):
    """Include true/false (t/f) positive/negative (p/n) as {tp,fp,tn,fn}"""
    tf_mul = np.zeros((len(hits),4))
    for edge, pred, label in zip(edges.T, preds, labels) :
        # True positives
        tf_mul[edge[0]][0] += int((pred > cut) and (label > cut))
        tf_mul[edge[1]][0] += int((pred > cut) and (label > cut))
        # False positives
        tf_mul[edge[0]][1] += int((pred > cut) and (label < cut))
        tf_mul[edge[1]][1] += int((pred > cut) and (label < cut))
        # True negatives
        tf_mul[edge[0]][2] += int((pred < cut) and (label < cut))
        tf_mul[edge[1]][2] += int((pred < cut) and (label < cut))
        # False negatives
        tf_mul[edge[0]][3] += int((pred < cut) and (label > cut))
        tf_mul[edge[0]][3] += int((pred < cut) and (label > cut))
        
    return tf_mul

def draw_triplets_tf_mul_xy(hits, edges, preds, labels, cut=0.5, figsize=(16, 16), lineWidth = 2 ):
    xi, yi = hits[:,0] * np.cos(hits[:,1]), hits[:,0] * np.sin(hits[:,1])
    xo, yo = hits[:,3] * np.cos(hits[:,4]), hits[:,3] * np.sin(hits[:,4])
    scores = hits[:,6]
    fig, ax0 = plt.subplots(figsize=figsize)
    
    tf_mul = tf_multiplicity(hits, edges, preds, labels, cut)
#     Draw the hits
    ax0.scatter(xi, yi, s=2, c='k')
    ax0.scatter(xo, yo, s=2, c='k')
    
    # Draw the segments
    for j in range(len(xi)):
        # Gold standard positives
        if (tf_mul[j][0]>0 and tf_mul[j][1]==0 
            and tf_mul[j][3]==0):
            ax0.plot([xi[j],xo[j]],[yi[j],yo[j]],
                     '-', c='#ffc70f', lineWidth= lineWidth)
        # Gold standards with missed tracks
        if (tf_mul[j][0]>0 and tf_mul[j][1]==0 
            and tf_mul[j][3]>0):
            ax0.plot([xi[j],xo[j]],[yi[j],yo[j]],
                     '--', c='#982e06')
            
        # Silver standard positives
        if (tf_mul[j][0]>0 and tf_mul[j][1]>0):
            ax0.plot([xi[j],xo[j]],[yi[j],yo[j]],
                     '-', c='#adadad')
        
        # Totally missed edges
        if (tf_mul[j][0]==0 and tf_mul[j][1]==0 and 
            tf_mul[j][3]>0):
            ax0.plot([xi[j],xo[j]],[yi[j],yo[j]],
                     '--', c='b')
            
        # Totally incorrect edges
        if (tf_mul[j][0]==0 and tf_mul[j][1]>0):
            ax0.plot([xi[j],xo[j]],[yi[j],yo[j]],
                     '-', c='r')
        
        # Weird things??
        if (tf_mul[j][0]==0 and tf_mul[j][1]>0 and 
           tf_mul[j][3] > 0):
            ax0.plot([xi[j],xo[j]],[yi[j],yo[j]],
                     ':', c='g')
    return fig, ax0