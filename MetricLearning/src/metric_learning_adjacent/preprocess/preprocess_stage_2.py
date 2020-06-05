import os
import time
import yaml
import pickle
import numpy as np
from random import shuffle, randint

from sklearn.neighbors import KDTree

import torch

from ..train_embed.utils_experiment import load_model as load_embed_model

if torch.cuda.is_available():
    DEVICE='cuda'
else:
    DEVICE='cpu'

RADIUS = 1.0

ALL_LAYERS = np.array([[8,2],
                       [8,4],
                       [8,6],
                       [8,8],
                       [13,2],
                       [13,4],
                       [13,6],
                       [13,8],
                       [17,2],
                       [17,4]])

def construct_dataset(paths, nb_samples, feature_names, emb_model):
    t0 = time.time()
    nb_processed = 0
    hits_a = []
    hits_b = []
    targets = []
    print("Sampling hit pairs for training dataset. \nWARNING: ASSUMING FIRST 3 FEATURES OF HITS ARE XYZ")
    emb_model = emb_model.to(DEVICE)
    emb_model.eval()
    for i, path in enumerate(paths):
        sample = load_event(path)
        hits, emb, particle_ids, vols, layers = process_sample(sample,
                                                               feature_names,
                                                               emb_model)
        h_a, h_b, t = build_pairs(hits, emb, particle_ids, vols, layers)
        hits_a.extend(h_a)
        hits_b.extend(h_b)
        targets.extend(t)
        if (i%2)==0:
            elapsed = (time.time() - t0)/60
            remain = (nb_samples-len(hits_a)) / len(hits_a) * elapsed
            print("file {:4d}, {:8d}. Elapsed: {:4.1f}m, Remain: {:4.1f}m".format(i,
                                         len(hits_a), elapsed, remain))
        if len(hits_a) > nb_samples:
            break
    return (hits_a[:nb_samples], hits_b[:nb_samples], targets[:nb_samples])

def process_emb(emb_model, hits):
    X = torch.FloatTensor(hits).to(DEVICE)
    with torch.autograd.no_grad():
        emb = emb_model(X)
    return emb.data.cpu().numpy()


def process_sample(sample, feature_names, emb_model):
    hits = sample[0]
    truth = sample[1]
    volume_ids = hits['volume_id'].values
    layer_ids  = hits['layer_id'].values
    hits = hits[feature_names].values.tolist()
    emb = process_emb(emb_model, hits)
    particle_ids = truth['particle_id'].values.tolist()
    return hits, emb, particle_ids, volume_ids, layer_ids

def get_dense_pairs(hits, where_track):
    hits_a = []
    hits_b = []
    len_track = len(where_track)
    for i in range(len_track):
        for j in range(len_track):
            hits_a.append(hits[where_track[i]])
            hits_b.append(hits[where_track[j]])
    return hits_a, hits_b

def is_match(hit_id_a, hit_id_b, vols, layers):
    va = vols[hit_id_a]
    vb = vols[hit_id_b]
    la = layers[hit_id_a]
    lb = layers[hit_id_b]

    for i, p in enumerate(ALL_LAYERS):
        if (p==[va, la]).all():
            if i==0:
                match_lower = False
            else:
                match_lower = ([vb,lb]==ALL_LAYERS[i-1]).all()
            if (i+1)==len(ALL_LAYERS):
                match_upper=False
            else:
                match_upper = ([vb,lb]==ALL_LAYERS[i+1]).all()

            if match_lower or match_upper:
                return True
    return False

def filter_by_radius(emb, idx_a, idx_b, distance_threshold):
    emb_a = emb[idx_a]
    emb_b = emb[idx_b]
    dist = np.sqrt(np.sum(np.square(emb_a - emb_b), axis=1)).tolist()
    ix_a = []
    ix_b = []
    for i, d in enumerate(dist):
        if d <= distance_threshold:
            ix_a.append(idx_a[i])
            ix_b.append(idx_b[i])
    return ix_a, ix_b

def get_true_pairs_layerwise(hits, emb, where_track, vols, layers):
    idx_a = []
    idx_b = []
    len_track = len(where_track)
    for i in range(len_track):
        for j in range((i+1), len_track):
            ha = where_track[i]
            hb = where_track[j]
            if is_match(ha, hb, vols, layers):
                idx_a.append(ha)
                idx_b.append(hb)

                idx_a.append(hb)
                idx_b.append(ha)
    idx_a, idx_b = filter_by_radius(emb, idx_a, idx_b, RADIUS)
    hits_a = [hits[idx] for idx in idx_a]
    hits_b = [hits[idx] for idx in idx_b]
    return hits_a, hits_b
    
# def get_true_pairs_layerwise(hits, where_track, z):
#     sorted_by_z = np.argsort(z[where_track]).tolist()
#     track_hits = [hits[i] for i in where_track]
#     track_hits = [track_hits[s] for s in sorted_by_z]
#     
#     hits_a = []
#     hits_b = []
#     len_track = len(where_track)
#     nb_processed = 0
#     for i in range(len_track):
#         lower_bound = i-min(1,nb_processed)
#         upper_bound = i+min(2,len_track-nb_processed)
#         for j in range(lower_bound, upper_bound):
#             hits_a.append(track_hits[i])
#             hits_b.append(track_hits[j])
#         nb_processed += 1
# 
#     return hits_a, hits_b
# 

def get_false_pairs(hits,
                    where_track,
                    neighbors_track,
                    particle_ids,
                    pid,
                    nb_false_pairs):
    idx_a = []
    idx_b = []
    max_track_idx = len(where_track) - 1
    count = 0
    while len(idx_a) < nb_false_pairs:
        count += 1
        i = randint(0, max_track_idx)
        seed_idx = where_track[i]
        neighbors = neighbors_track[i]
        neighbor_idx = neighbors[randint(0, len(neighbors) - 1)]

        if particle_ids[seed_idx] == particle_ids[neighbor_idx]:
            continue

        idx_a.append(seed_idx)
        idx_b.append(neighbor_idx)

        if count > 3 * nb_false_pairs:
            print("could not get nb false pairs requested")
            break

    h_a = []
    h_b = []
    for i,j in zip(idx_a, idx_b):
        h_a.append(hits[i])
        h_b.append(hits[j])
    return h_a, h_b

def get_pairs_one_pid(hits, emb, tree, particle_ids, pid, z, vols, layers):
    where_track = list(np.where(particle_ids==pid)[0])
    emb_track = emb[where_track]
    neighbors_track = tree.query_radius(emb_track, r=RADIUS)

    # h_true_a, h_true_b = get_dense_pairs(hits, where_track)
    h_true_a, h_true_b = get_true_pairs_layerwise(hits,
                                                  emb,
                                                  where_track,
                                                  vols,
                                                  layers)
    target_true = [1] * len(h_true_a)

    if len(h_true_a)==0:
        return [], [], []

    h_false_a, h_false_b = get_false_pairs(hits,
                                           where_track,
                                           neighbors_track,
                                           particle_ids,
                                           pid,
                                           len(h_true_a))
    target_false = [0] * len(h_false_a)

    return h_true_a+h_false_a, h_true_b+h_false_b, target_true+target_false

def build_pairs(hits,
                emb,
                particle_ids,
                vols,
                layers,
                nb_particles_per_sample=2000):
    unique_pids = list(set(particle_ids))
    pids = np.array(particle_ids)
    shuffle(unique_pids)
    hits_a = []
    hits_b = []
    target  = []
    tree = KDTree(emb)
    z = np.array(hits)[:,2]
    for i in range(nb_particles_per_sample):
        pid = unique_pids[i]
        h_a, h_b, t = get_pairs_one_pid(hits,
                                        emb,
                                        tree,
                                        pids,
                                        pid,
                                        z,
                                        vols,
                                        layers)
        hits_a.extend(h_a)
        hits_b.extend(h_b)
        target.extend(t)
    return hits_a, hits_b, target

def combine_samples(hits_a, hits_b, targets):
    t = np.array(targets, dtype=np.float32).reshape(-1,1)
    return np.concatenate((hits_a, hits_b, t),axis=1).astype(np.float32)

def preprocess_dataset(paths, nb_samples, feature_names, emb_model):
    h_a, h_b, t = construct_dataset(paths, nb_samples, feature_names, emb_model)
    dataset = combine_samples(h_a, h_b, t)
    mean, std = extract_stats(h_a, h_b)
    stats = {'mean':mean, 'std':std}
    return dataset, stats

def extract_stats(h_a, h_b):
    h_a = np.array(h_a, dtype=np.float32)
    h_b = np.array(h_b, dtype=np.float32)
    h_combined = np.concatenate((h_a,h_b),axis=0)
    mean = np.mean(h_combined,axis=0)
    std  = np.std(h_combined, axis=0)
    return mean, std

#############################################
#                   UTILS                   #
#############################################
def save_stats(stats, save_path, name):
    save_file = os.path.join(save_path, "{}.yml".format('stats'))
    with open(save_file, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)

def save_dataset(dataset, save_path, name):
    save_file = os.path.join(save_path, "{}.pickle".format(name))
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f)

def load_event(path):
    with open(path, 'rb') as f:
        sample = pickle.load(f)
    return sample

def split_dataset(dataset, nb_train, nb_valid, nb_test):
    assert len(dataset) >= (nb_train + nb_valid + nb_test)
    np.random.shuffle(dataset)
    train = dataset[:nb_train]
    valid = dataset[nb_train:(nb_train+nb_valid)]
    test  = dataset[-nb_test:]
    return train, valid, test

#############################################
#                   MAIN                    #
#############################################
def preprocess(artifact_storage,
               data_path,
               save_dir,
               feature_names,
               nb_train,
               nb_valid,
               nb_test,
               force=False):
    if os.path.isdir(save_dir) and (force!=True):
        print("Stage 2 preprocessing dir exists")
    elif os.path.isfile(os.path.join(artifact_storage, 'metric_learning_filter', 'best_model.pkl')) and (force!=True):
        print("Best filter model exists from previous run. Not forcing preprocessing stage 2.")
    else:
        event_files = os.listdir(data_path)
        event_paths = [os.path.join(data_path, f) for f in event_files]
        shuffle(event_paths)

        nb_samples = nb_train + nb_valid + nb_test
        
        best_emb_path = os.path.join(artifact_storage, 'metric_learning_emb', 'best_model.pkl')
        emb_model = load_embed_model(best_emb_path, DEVICE)
    
        dataset, stats = preprocess_dataset(event_paths,
                                            nb_samples,
                                            feature_names,
                                            emb_model)

        os.makedirs(save_dir, exist_ok=True)
        save_stats(stats, save_dir, 'stage_2')
        train, valid, test = split_dataset(dataset, nb_train, nb_valid, nb_test)
        save_dataset(train, save_dir, 'train')
        save_dataset(valid, save_dir, 'valid')
        save_dataset(test, save_dir, 'test')
    return save_dir

def main(args, force=False):
    
    save_path = os.path.join(args.data_storage_path, 'metric_stage_2')
    load_path = os.path.join(args.data_storage_path, 'preprocess_raw')
    
    preprocess(args.artifact_storage_path,
               load_path,
               save_path,
               args.feature_names,
               args.nb_train,
               args.nb_valid,
               args.nb_test,
               force)

if __name__ == "__main__":
    args = read_args()
    main(args)
