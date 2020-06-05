import os
import time
import yaml
import pickle
import numpy as np
from random import shuffle

from sklearn.neighbors import KDTree

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

def construct_dataset(paths, nb_samples, feature_names):
    t0 = time.time()
    nb_processed = 0
    hits_a = []
    hits_b = []
    targets = []
    print("Sampling hit pairs for training dataset. \nWARNING: ASSUMING FIRST 3 FEATURES OF HITS ARE XYZ")
    for i, path in enumerate(paths):
        sample = load_event(path)
        hits, particle_ids, vols, layers = process_sample(sample, feature_names)
        h_a, h_b, t = build_pairs(hits, particle_ids, vols, layers)
        hits_a.extend(h_a)
        hits_b.extend(h_b)
        targets.extend(t)
        if (i%2)==0:
            elapsed = (time.time() - t0)/60
            remain = (nb_samples-len(hits_a)) / len(hits_a) * elapsed # THIS ALGORITHM IS OFF??
            print("file {:4d}, {:8d}. Elapsed: {:4.1f}m, Remain: {:4.1f}m".format(i,
                                         len(hits_a), elapsed, remain))
        if len(hits_a) > nb_samples:
            break
    return (hits_a[:nb_samples], hits_b[:nb_samples], targets[:nb_samples])

def process_sample(sample, feature_names):
    hits = sample[0]
    truth = sample[1]
    volume_ids = hits['volume_id'].values
    layer_ids  = hits['layer_id'].values
    hits = hits[feature_names].values.tolist()
    particle_ids = truth['particle_id'].values.tolist()
    return hits, particle_ids, volume_ids, layer_ids

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

def get_true_pairs_layerwise(hits, where_track, vols, layers):
    hits_a = []
    hits_b = []
    len_track = len(where_track)
    for i in range(len_track):
        for j in range((i+1), len_track):
            ha = where_track[i]
            hb = where_track[j]
            if is_match(ha, hb, vols, layers):
                hits_a.append(hits[ha])
                hits_b.append(hits[hb])

                hits_a.append(hits[hb])
                hits_b.append(hits[ha])
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

def get_false_pairs(hits, where_track, particle_ids, pid, nb_false_pairs):
    h_a = []
    h_b = []
    where_not_track = np.where(particle_ids!=pid)[0]
    where_not_track = list(np.random.choice(where_not_track, nb_false_pairs, replace=False))
    seed_hit = hits[where_track[np.random.randint(len(where_track))]]

    track_hit_order = list(np.random.choice(where_track, nb_false_pairs))

    for i,j in zip(track_hit_order, where_not_track):
        h_a.append(hits[i])
        h_b.append(hits[j])
    return h_a, h_b

def get_pairs_one_pid(hits, particle_ids, pid, z, vols, layers):
    where_track = list(np.where(particle_ids==pid)[0])

    # h_true_a, h_true_b = get_dense_pairs(hits, where_track)
    h_true_a, h_true_b = get_true_pairs_layerwise(hits, where_track, vols, layers)
    target_true = [1] * len(h_true_a)

    if len(h_true_a)==0:
        return [], [], []

    h_false_a, h_false_b = get_false_pairs(hits, where_track, particle_ids, pid, len(h_true_a))
    target_false = [0] * len(h_false_a)

    return h_true_a+h_false_a, h_true_b+h_false_b, target_true+target_false

def build_pairs(hits, particle_ids, vols, layers, nb_particles_per_sample=2000):
    unique_pids = list(set(particle_ids))
    pids = np.array(particle_ids)
    shuffle(unique_pids)
    hits_a = []
    hits_b = []
    target  = []
    z = np.array(hits)[:,2]
    for i in range(nb_particles_per_sample):
        pid = unique_pids[i]
        h_a, h_b, t = get_pairs_one_pid(hits, pids, pid, z, vols, layers)
        hits_a.extend(h_a)
        hits_b.extend(h_b)
        target.extend(t)
    return hits_a, hits_b, target

def combine_samples(hits_a, hits_b, targets):
    t = np.array(targets, dtype=np.float32).reshape(-1,1)
    return np.concatenate((hits_a, hits_b, t),axis=1).astype(np.float32)

def preprocess_dataset(paths, nb_samples, feature_names):
    h_a, h_b, t = construct_dataset(paths, nb_samples, feature_names)
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
    print(len(dataset), nb_train, nb_valid, nb_test)
    assert len(dataset) >= (nb_train + nb_valid + nb_test)
    np.random.shuffle(dataset)
    train = dataset[:nb_train]
    valid = dataset[nb_train:(nb_train+nb_valid)]
    test  = dataset[-nb_test:]
    return train, valid, test

#############################################
#                   MAIN                    #
#############################################
def preprocess(experiment_name, artifact_storage, data_path, feature_names, save_dir, nb_train, nb_valid, nb_test, force=False):
    if os.path.isdir(save_dir) and (force!=True):
        print("Stage 1 preprocessing dir exists")
    elif os.path.isfile(os.path.join(artifact_storage, 'metric_learning_emb', 'best_model.pkl')) and (force!=True):
        print("Best embedding model exists from previous run. Not forcing preprocessing stage 1.")
    else:
        print("Saving to:",str(os.path.join(artifact_storage, experiment_name, 'metric_learning_emb', 'best_model.pkl')))
        event_files = os.listdir(data_path)
        event_paths = [os.path.join(data_path, f) for f in event_files]
        shuffle(event_paths)

        nb_samples = nb_train + nb_valid + nb_test
        dataset, stats = preprocess_dataset(event_paths, nb_samples, feature_names)

        os.makedirs(save_dir, exist_ok=True)
        save_stats(stats, save_dir, 'stage_1')
        train, valid, test = split_dataset(dataset, nb_train, nb_valid, nb_test)
        save_dataset(train, save_dir, 'train')
        save_dataset(valid, save_dir, 'valid')
        save_dataset(test, save_dir, 'test')
    return save_dir

def main(args, force=False):
    
    save_path = os.path.join(args.data_storage_path, 'metric_stage_1')
    load_path = os.path.join(args.data_storage_path, 'preprocess_raw')
    
    preprocess(args.name,
               args.artifact_storage_path,
               load_path,
               args.feature_names,
               save_path,
               args.nb_train,
               args.nb_valid,
               args.nb_test,
               force)

if __name__ == "__main__":
    
    args = read_args()
    main(args)
