import os
import time
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import functools

import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool 
from multiprocessing import Pool as ProcessPool 

import trackml.dataset

from .extract_dir import extract_dir

BARREL_VOLUMES = [8, 13, 17]

#############################################
#               REDUCE TRACKS               #
#############################################
def remove_all_noise(hits, cells, truth):
    unique_ids = truth.particle_id.unique()
    track_ids_to_keep = unique_ids[np.where(unique_ids != 0)]

    where_to_keep = truth['particle_id'].isin(track_ids_to_keep)

    hits_reduced  = hits[where_to_keep]
    truth_reduced = truth[where_to_keep]

    hit_ids_to_keep = truth_reduced.hit_id.values
    cells_reduced = cells[cells['hit_id'].isin(hit_ids_to_keep)]
    return hits_reduced, cells_reduced, truth_reduced

def remove_all_endcaps(hits, cells, truth):
    where_to_keep = hits['volume_id'].isin(BARREL_VOLUMES)

    hits_reduced  = hits[where_to_keep]
    truth_reduced = truth[where_to_keep]

    hit_ids_to_keep = truth_reduced.hit_id.values
    cells_reduced = cells[cells['hit_id'].isin(hit_ids_to_keep)]
    return hits_reduced, cells_reduced, truth_reduced

def apply_pt_cut(hits, truth, cells, pt_cut=0):
    
    hits_reduced = hits[truth.pt > pt_cut]
    truth_reduced = truth[truth.pt > pt_cut]
    cells_reduced = cells[cells.hit_id.isin(hits_reduced.hit_id)]
    
    return hits_reduced, truth_reduced, cells_reduced
    


#############################################
#           FEATURE_AUGMENTATION            #
#############################################
def augment_hit_features(hits, cells, detector_orig, detector_proc):

    cell_stats = get_cell_stats(cells)
    hits['cell_count'] = cell_stats[:,0]
    hits['cell_val']   = cell_stats[:,1]
    
    hits = extract_dir(hits, cells, detector_orig, detector_proc)

    return hits

def get_cell_stats(cells):
    hit_cells = cells.groupby(['hit_id']).value.count().values
    hit_value = cells.groupby(['hit_id']).value.sum().values
    cell_stats = np.hstack((hit_cells.reshape(-1,1), hit_value.reshape(-1,1)))
    cell_stats = cell_stats.astype(np.float32)
    return cell_stats
    
#############################################
#               DETECTOR UTILS              #
#############################################
def load_detector(detector_path):
    detector_orig = pd.read_csv(detector_path)
    detector_pfx = detector_path.split('.')[0]
    detector_preproc = detector_pfx + ".pickle"
    try:
        print("Loading detector...")
        with open(detector_preproc, 'rb') as f:
            detector = pickle.load(f)
        print("Detector loaded.")
    except:
        print("Failed to load preprocessed detector. Building...")
        detector = pd.read_csv(detector_path)
        detector = preprocess_detector(detector)
        with open(detector_preproc, 'xb') as f:
            pickle.dump(detector, f)
        print("Detector preprocessed and saved.")
    return detector_orig, detector

def preprocess_detector(detector):
    thicknesses = Detector_Thicknesses(detector).get_thicknesses()
    rotations = Detector_Rotations(detector).get_rotations()
    pixel_size = Detector_Pixel_Size(detector).get_pixel_size()
    det = dict(thicknesses=thicknesses,
               rotations=rotations,
               pixel_size=pixel_size)
    return det


def determine_array_size(detector):
    max_v, max_l, max_m = (0,0,0)
    unique_vols = detector.volume_id.unique()
    max_v = max(unique_vols)+1
    for v in unique_vols:
        vol = detector.loc[detector['volume_id']==v]
        unique_layers = vol.layer_id.unique()
        max_l = max(max_l, max(unique_layers)+1)
        for l in unique_layers:
            lay = vol.loc[vol['layer_id']==l]
            unique_modules = lay.module_id.unique()
            max_m = max(max_m, max(unique_modules)+1)
    return max_v, max_l, max_m

class Detector_Rotations(object):
    def __init__(self, detector):
        self.detector = detector
        self.max_v, self.max_l, self.max_m = determine_array_size(detector)

    def get_rotations(self):
        print("  Extracting rotations...")
        self._init_rotation_array()
        self._extract_all_rotations()
        print("  Done.")
        return self.rot

    def _init_rotation_array(self):
        self.rot =  np.zeros((self.max_v, self.max_l, self.max_m, 3, 3))

    def _extract_all_rotations(self):
        for i, r in self.detector.iterrows():
            v, l, m = tuple(map(int, (r.volume_id, r.layer_id, r.module_id)))
            rot = self._extract_rotation_matrix(r)
            self.rot[v, l, m] = rot
            
    def _extract_rotation_matrix(self, mod) :
      '''
      Extract the rotation matrix from module dataframe
      '''
      r = np.matrix([[ mod.rot_xu.item(),mod.rot_xv.item(),mod.rot_xw.item()],
                     [mod.rot_yu.item(),mod.rot_yv.item(),mod.rot_yw.item()],
                     [mod.rot_zu.item(),mod.rot_zv.item(),mod.rot_zw.item()]])
      return r

class Detector_Thicknesses(object):
    def __init__(self, detector):
        self.detector = detector
        self.max_v, self.max_l, self.max_m = determine_array_size(detector)

    def get_thicknesses(self):
        print("  Extracting thicknesses...")
        self._init_thickness_array()
        self._extract_all_thicknesses()
        print("  Done.")
        return self.all_t

    def _init_thickness_array(self):
        self.all_t =  np.zeros((self.max_v, self.max_l, self.max_m))

    def _extract_all_thicknesses(self):
        for i, r in self.detector.iterrows():
            v, l, m = tuple(map(int, (r.volume_id, r.layer_id, r.module_id)))
            self.all_t[v, l, m] = r.module_t

class Detector_Pixel_Size(object):
    def __init__(self, detector):
        print(detector.keys())
        self.detector = detector
        self.max_v, self.max_l, self.max_m = determine_array_size(detector)

    def get_pixel_size(self):
        print("  Extracting thicknesses...")
        self._init_size_array()
        self._extract_all_size()
        print("  Done.")
        return self.all_s

    def _init_size_array(self):
        self.all_s =  np.zeros((self.max_v, self.max_l, self.max_m, 2))

    def _extract_all_size(self):
        for i, r in self.detector.iterrows():
            v, l, m = tuple(map(int, (r.volume_id, r.layer_id, r.module_id)))
            self.all_s[v, l, m, 0] = r.pitch_u
            self.all_s[v, l, m, 1] = r.pitch_v


###########################################
#               EVENT LOADING             #
###########################################


def get_one_event(event_path,
                  detector_orig,
                  detector_proc,
                  include_endcaps,
                  include_noise,
                  pt_cut):

    print("Loading event {} with a {} pT cut".format(event_path, pt_cut))
    hits, cells, particles, truth = trackml.dataset.load_event(event_path)
    pt = np.sqrt(particles.px**2 + particles.py**2 + particles.pz**2)
    particles = particles.assign(pt=pt)

    if not include_noise:
        hits, cells, truth = remove_all_noise(hits, cells, truth)

    if not include_endcaps:
        hits, cells, truth = remove_all_endcaps(hits, cells, truth)
    truth = truth.merge(particles[['particle_id', 'pt']], on='particle_id')
    truth = truth.sort_values(by='hit_id')
    truth = truth.set_index([truth['hit_id']-1])
    hits, truth, cells = apply_pt_cut(hits, truth, cells, pt_cut)

    try:
        hits = augment_hit_features(hits, cells, detector_orig, detector_proc)
    except Exception as e:
        print(e)
        print("Number hits:  {}".format(hits.shape))
        print("Number cells: {}".format(cells.shape))
        raise Exception("Error augmenting hits.")
    return [hits, truth]

    
#########################################################
#               PICKLE DATASET LOAD / SAVE              #
#########################################################
def get_event_paths(data_path, data_dir_name):
    train_dir = os.path.join(data_path, data_dir_name)
    event_names = [e.split('-')[0] for e in os.listdir(train_dir)]
    event_names = list(set(event_names))
    event_names.sort()
    event_paths = [os.path.join(train_dir, e) for e in event_names]
    return event_paths


def save_preprocessed_event(event, preproc_path, preproc_file):
    os.makedirs(preproc_path, exist_ok=True)
    with open(preproc_file, 'wb') as f:
        pickle.dump(event, f)

def preprocess_one_event(event_path,
                         preproc_path,
                         percent_keep,
                         detector_orig,
                         detector_proc,
                         include_endcaps,
                         include_noise,
                         pt_cut,
                         force):
    i, event_path = event_path
    if (i%10)==0:
        print("{:5d}".format(i))
    event_name = event_path.split('/')[-1]
    print(event_name)
    preproc_file = os.path.join(preproc_path, "{}.pickle".format(event_name))
    if not os.path.exists(preproc_file) or force:
        try:
            event = get_one_event(event_path,
                                  detector_orig,
                                  detector_proc,
                                  include_endcaps,
                                  include_noise,
                                  pt_cut)
            save_preprocessed_event(event, preproc_path, preproc_file)
        except Exception as e:
            print(e)
            exit()
    else:
        print("File already exists at {}".format(preproc_file))

def preprocess_with_threads(event_paths, preproc_path, percent_keep, detector, pt_cut):
    func = functools.partial(preprocess_one_event,
                             preproc_path=preproc_path,
                             percent_keep=percent_keep,
                             detector=detector,
                             pt_cut=pt_cut)
    if percent_keep < 0.05:
        print("Using two threads")
        pool = ThreadPool(2) 
        events = pool.map(func, event_paths)
    else:
        print("Using one thread")
        for e in event_paths:
            func(e)

def preprocess_with_processes(event_paths,
                              preproc_path,
                              percent_keep,
                              detector_orig,
                              detector_proc,
                              include_endcaps,
                              include_noise,
                              pt_cut,
                              force):
    event_paths = [(i,e) for i,e in enumerate(event_paths)]
    func = functools.partial(preprocess_one_event,
                             preproc_path=preproc_path,
                             percent_keep=percent_keep,
                             detector_orig=detector_orig,
                             detector_proc=detector_proc,
                             include_endcaps=include_endcaps,
                             include_noise=include_noise,
                             pt_cut=pt_cut,
                             force=force)
    pool = ProcessPool(multiprocessing.cpu_count()) 
    events = pool.map(func, event_paths)
    
    
def preprocess_dataset(data_path,
                       save_path,
                       detector_path,
                       pt_cut,
                       task,
                       n_tasks,
                       data_dir_name,
                       include_endcaps=True,
                       include_noise=True,
                       force=False):
    '''
    Preprocess one dataset folder.
    Keep only percent_keep tracks.
    '''
    percent_keep=1.0
    cpu_count = multiprocessing.cpu_count()
    detector_orig, detector_proc = load_detector(detector_path)
    event_paths = get_event_paths(data_path, data_dir_name)
    
    # Split the files into n_tasks and select the ith split	
    task_event_paths = np.array_split(event_paths, n_tasks)[task]
    
    print("Preprocessing all events")
    print("Using {} cpu cores".format(cpu_count))
    t0 = time.time()
    if cpu_count == 1:
        preprocess_with_threads(task_event_paths, save_path, percent_keep, detector, pt_cut, force)
    else:
        preprocess_with_processes(task_event_paths,
                                  save_path,
                                  percent_keep,
                                  detector_orig,
                                  detector_proc,
                                  include_endcaps,
                                  include_noise,
                                  pt_cut,
                                  force)
    t1 = time.time()
    print("Preprocessing finished")
    print("Required {:.2f} minutes for {} files".format((t1-t0)/60, len(event_paths)))
    print("{:.4f} seconds per file".format(cpu_count*(t1-t0) / len(event_paths)))

def main(args,
         force=False,
         verbose=False):
    save_path = os.path.join(args.data_storage_path, 'preprocess_raw')
    if os.path.isdir(save_path) and (not force):
        print("Raw data path exists for experiment", args.name)
    else:
        print("Processing raw data for experiment", args.name)
        raw_subdirs = [ name for name in os.listdir(args.raw_data_path) if os.path.isdir(os.path.join(args.raw_data_path, name)) ]
        
        """
        Handle either a given directory of data, or a specified number of subdirectories of data
        """
        if len(raw_subdirs) == 0:
            preprocess_dataset(args.raw_data_path,
                               save_path,
                               args.detector_path,
                               args.pt_cut,
                               args.task,
                               args.n_tasks,
                               "",
                               args.include_endcaps,
                               args.include_noise,
                               force)
        else:        
            nb_processed=0
            for raw_name in raw_subdirs:
                preprocess_dataset(args.raw_data_path,
                                   save_path,
                                   args.detector_path,
                                   args.pt_cut,
                                   args.task,
                                   args.n_tasks,
                                   raw_name,
                                   args.include_endcaps,
                                   args.include_noise,
                                   force)
                nb_processed+=1
                if args.nb_to_process==-1:
                    continue
                elif nb_processed == args.nb_to_process:
                    break
    feature_names = ['x','y','z','cell_count', 'cell_val',
                     'leta', 'lphi', 'lx', 'ly', 'lz', 'geta', 'gphi']
    return save_path, feature_names
