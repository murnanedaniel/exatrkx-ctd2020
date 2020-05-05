import os
import yaml
import pickle
import numpy as np

from cycler import cycler
import matplotlib; matplotlib.use('Agg') # no display on clusters
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

import torch

from sklearn.decomposition import TruncatedSVD

import model

#################################
#               3D              #
#################################

def plot_3d(hits, particle_ids, savedir, name):
    fig = plt.figure()
    ax = plt.gca(projection='3d',zorder=1)
    ax.set_prop_cycle(cycler('color', ['b', 'g', 'r', 'c', 'm', 'y', 'k']))
    ax.set_title(name)
    unique_particles = np.unique(particle_ids)
    for i, pid in enumerate(unique_particles):
        h_idx = np.where(particle_ids==pid)
        h = hits[h_idx]
        ax.scatter3D(h[:,0], h[:,1], h[:,2],zorder=3)
        if i == 6:
            break
    fig.savefig(os.path.join(savedir, name), dpi=200)
    plt.close(fig)

def visualize_event3d(net, norm_stats, hits, truth, savedir, idx):
    particle_ids = truth['particle_id'].values
    hits = hits[['x','y','z','x_dir','y_dir','z_dir']].values

    plot_3d(hits[:,:3],particle_ids, savedir, str(idx)+"_3Da_orig.png")

    net.eval()
    with torch.autograd.no_grad():
        h = (hits-norm_stats['mean'])/norm_stats['std']
        emb_hits = net(torch.Tensor(h)).data.numpy()
    plot_3d(emb_hits,particle_ids, savedir, str(idx)+"_3Db_emb.png")

#################################
#               2D              #
#################################
def plot_2d(hits, particle_ids, savedir, name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(cycler('color', ['b', 'g', 'r', 'c', 'm', 'y', 'k']))
    ax.set_title(name)
    unique_particles = np.unique(particle_ids)
    for i, pid in enumerate(unique_particles):
        h_idx = np.where(particle_ids==pid)
        h = hits[h_idx]
        ax.scatter(h[:,0], h[:,1])
        if i == 6:
            break
    fig.savefig(os.path.join(savedir, name), dpi=200)
    plt.close(fig)

def visualize_event2d(net, norm_stats, hits, truth, savedir, idx):
    particle_ids = truth['particle_id'].values
    hits = hits[['x','y','z','x_dir','y_dir','z_dir']].values

    '''
    plot_2d(hits[:,:2],particle_ids, savedir, str(idx)+"_a_orig.png")

    svd = TruncatedSVD(n_components=2)
    h = svd.fit_transform(hits)
    plot_2d(h,particle_ids, savedir, str(idx)+"_c_svd.png")
    '''

    net.eval()
    with torch.autograd.no_grad():
        h = (hits-norm_stats['mean'])/norm_stats['std']
        emb_hits = net(torch.Tensor(h)).data.numpy()
    plot_3d(emb_hits,particle_ids, savedir, str(idx)+"_3Db_s2emb.png")

def visualize_graph(data_dir, model_filepath, norm_file, nb_plot, savedir):
    event_filepaths = [os.path.join(data_dir, e) for e in os.listdir(data_dir)]
    net = torch.load(model_filepath)
    norm_stats = yaml.load(open(norm_file,'r'))
    for i in range(nb_plot):
        print("Plotting {}...".format(i))
        with open(event_filepaths[i],'rb') as f:
            hits, truth = pickle.load(f)
        visualize_event2d(net, norm_stats, hits, truth, savedir, i)
        print("Done.")
    

if __name__ == "__main__":
    '''
    visualize_graph('/misc/vlgscratch4/BrunaGroup/choma/track_ml/preprocess/keep_0.0010',
                    '/home/nc2201/research/trackML_all/trackML/experiments/metric_learning/models/ea_hinge_3d/0/best_model.pkl',
                    '/misc/vlgscratch4/BrunaGroup/choma/track_ml/preprocess_metric/stage_one.yml',
                    4,
                    '/home/nc2201/research/trackML_all/trackML/experiments/metric_learning/models/vis',
                    )
    '''
    visualize_graph('/misc/vlgscratch4/BrunaGroup/choma/track_ml/preprocess/keep_0.0010',
                    '/home/nc2201/research/trackML_all/trackML/experiments/metric_learning/models/fa_stage_ii_3d/0/best_model.pkl',
                    '/misc/vlgscratch4/BrunaGroup/choma/track_ml/preprocess_metric_ii/stage_ii.yml',
                    4,
                    '/home/nc2201/research/trackML_all/trackML/experiments/metric_learning/models/vis',
                    )
