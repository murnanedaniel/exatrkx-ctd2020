import os
import sys
import csv
import time
import argparse
import logging
import pickle
import yaml
import numpy as np
import pandas as pd
from random import shuffle

from sklearn.metrics import roc_auc_score, roc_curve

from trackml.score import score_event
import trackml.dataset

# import matplotlib; matplotlib.use('Agg') # no display on clusters
# import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

from . import model

#####################
#     CONSTANTS     #
#####################
ARGS_NAME  = 'args.yml'
MODEL_NAME = 'model.pkl'
BEST_MODEL = 'best_model.pkl'
STATS_CSV  = 'training_stats.csv'
TEST_STATS_FILE= 'test_stats.yml'
MODELS_DIR = 'models'
PLOT_DIR   = 'plots'
SPATIAL_DIMS = [0,1,2]

#####################################
#     EXPERIMENT INITIALIZATION     #
#####################################

def read_args():

  parser = argparse.ArgumentParser(description=
                      'Arguments for GNN model and experiment')
  add_arg = parser.add_argument

  # Experiment
  add_arg('--name', help='Experiment reference name', required=True)
  add_arg('--run', help='Experiment run number', type=int, default=0)
  add_arg('--evaluate', help='Evaluate test set only', action='store_true')
  add_arg('--train_data', help='Train data location',required=True)
  add_arg('--valid_data', help='Valid data location',required=True)
  add_arg('--test_data', help='Test data location',required=True)
  add_arg('--stats_path', help='Location of dataset mean, std',required=True)

  # Training
  add_arg('--max_nb_epoch', help='Max nb epochs to train', type=int, default=2)
  add_arg('--lr_start', help='Initial learn rate', type=float, default = 0.005)
  add_arg('--lr_end', help='Minimum LR', type=float, default = 10**-6)
  add_arg('--batch_size', help='Size of each minibatch', type=int, default=1)

  # Dataset
  add_arg('--nb_train', help='Number of train samples', type=int, default=10)
  add_arg('--nb_valid', help='Number of valid samples', type=int, default=10)
  add_arg('--nb_test', help='Number of test samples', type=int, default=10)

  # Model Architecture
  add_arg('--input_dim', help='# input features', type=int, default=10)
  add_arg('--nb_hidden', help='# hidden units per layer', type=int, default=32)
  add_arg('--nb_layer', help='# network graph conv layers', type=int, default=6)
  add_arg('--emb_dim', help='Dimension to embed into', type=int, default=3)


  return parser.parse_args()

def initialize_logger(experiment_dir):
  '''
  Logger prints to stdout and logfile
  '''
  logfile = os.path.join(experiment_dir, 'log.txt')
  logging.basicConfig(filename=logfile,format='%(message)s',level=logging.INFO)
  logging.getLogger().addHandler(logging.StreamHandler())

def get_experiment_dir(experiment_name, run_number):
  current_dir = os.getcwd()
  save_dir = os.path.join(current_dir, MODELS_DIR)
  if not os.path.exists(save_dir):
    os.mkdir(save_dir) # Create models dir which will contain experiment data
  return os.path.join(save_dir, experiment_name, str(run_number))

def initialize_experiment_if_needed(model_dir, evaluate_only):
  '''
  Check if experiment initialized and initialize if not.
  Perform evaluate safety check.
  '''
  if not os.path.exists(model_dir):
    initialize_experiment(model_dir)
    if evaluate_only:
      logging.warning("EVALUATING ON UNTRAINED NETWORK")

def initialize_experiment(experiment_dir):
  '''
  Create experiment directory and initiate csv where epoch info will be stored.
  '''
  print("Initializing experiment.")
  os.makedirs(experiment_dir)
  csv_path = os.path.join(experiment_dir, STATS_CSV)
  with open(csv_path, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Epoch', 'lrate', 'train_score', 'train_loss', 'val_score', 'val_loss'])


###########################
#     MODEL UTILITIES     #
###########################
def create_or_restore_model(
                            experiment_dir,
                            input_dim,
                            nb_hidden,
                            nb_layer,
                            emb_dim,
                            stats_path
                            ):
  '''
  Checks if model exists and creates it if not.
  Returns model.
  '''
  model_file = os.path.join(experiment_dir, MODEL_NAME)
  if os.path.exists(model_file):
    logging.warning("Loading model...")
    m = load_model(model_file)
    logging.warning("Model restored.")
  else:
    logging.warning("Creating new model:")
    mean, std = load_stats(stats_path)
    m = model.MLP(nb_hidden, nb_layer, input_dim, mean, std, emb_dim)
    logging.info(m)
    save_model(m, model_file)
    logging.warning("Initial model saved.")
  return m

def load_stats(stats_path):
    with open(stats_path, 'r') as f:
        stats = yaml.load(f, Loader=yaml.Loader)
    return stats['mean'], stats['std']

def load_model(model_file, device='cpu'):
  '''
  Load torch model.
  '''
  m = torch.load(model_file, map_location=device)
  return m

def load_best_model(experiment_dir):
  '''
  Load the model which performed best in training.
  '''
  best_model_path = os.path.join(experiment_dir, BEST_MODEL)
  return load_model(best_model_path)

def save_model(m, model_file):
  '''
  Save torch model.
  '''
  torch.save(m, model_file)

def save_best_model(experiment_dir, net):
  '''
  Called if current model performs best.
  '''
  model_path = os.path.join(experiment_dir, BEST_MODEL)
  _save_model(model_path, net)
  logging.warning("Best model saved.")

def save_epoch_model(experiment_dir, net):
  '''
  Optionally called after each epoch to save current model.
  '''
  model_path = os.path.join(experiment_dir, MODEL_NAME)
  _save_model(model_path, net)
  logging.warning("Current model saved.")

def _save_model(savepath, net):
  '''
  Save model to savepath.
  '''
  save_model(net.cpu(), savepath)
  # Place network back on GPU if necessary
  if torch.cuda.is_available():
    net.cuda()


def load_args(experiment_dir):
  args_path = os.path.join(experiment_dir, ARGS_NAME)
  with open(args_path, 'r') as argfile:
    args = yaml.load(argfile)
  logging.warning("Model arguments restored.")
  return args

def save_args(experiment_dir, args):
  args_path = os.path.join(experiment_dir, ARGS_NAME)
  with open(args_path, 'w') as argfile:
    yaml.dump(args, argfile, default_flow_style=False)
  logging.warning("Experiment arguments saved")

#########################################
#               TRAINING                #
#########################################
class PG_Tracker(object):
    def __init__(self, decay_factor):
        self.dec_factor = decay_factor

    def get_reward(self, score):
        try:
            avg_score = avg_score * self.dec_factor + score*(1-self.dec_factor)
        except:
            avg_score = score
        return score - avg_score
            
            

######################
#     EVALUATION     #
######################
def score_dist_accuracy(pred, true):
    pred = pred.round()
    pred[pred!=0] = 1
    pred = 1-pred
    correct = pred==true
    nb_correct = correct.sum()
    nb_total = true.size(0)
    score = float(nb_correct.item()) / nb_total
    # print(nb_correct.item(), nb_total, score)
    return score

def track_epoch_stats(epoch_nb, lrate, train_stats, val_stats, experiment_dir):
  csv_path = os.path.join(experiment_dir, STATS_CSV)
  with open(csv_path, 'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([epoch_nb, 
                     lrate,
                     train_stats[0],
                     train_stats[1],
                     val_stats[0],
                     val_stats[1]])

def save_test_stats(experiment_dir, test_stats):
    stats = {'loss':test_stats[0],
             'dist_accuracy':test_stats[1]}
    stats_file = os.path.join(experiment_dir, TEST_STATS_FILE)
    with open(stats_file, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)


####################
#     PRINTING     #
####################
def print_header():
  '''
  Print header before train / evaluation run.
  '''
  logging.info("         Loss  Score")

def print_eval_stats(nb_processed, loss, score):
  '''
  Log stats during train, evaluation run.
  '''
  logging.info("  {:5d}: {:.3f}  {:2.2f}".format(nb_processed, loss, score))

#########################
#     VISUALIZATION     #
#########################
# def plot_roc(event_id, experiment_dir, y_true, y_pred, weights):
#   '''
#   Plot ROC curve for a given event.
#   '''
#   # Get plot output path and name
#   pathout = os.path.join(experiment_dir, PLOT_DIR)
#   # Make plots directory if needed
#   if not os.path.exists(pathout):
#     os.mkdir(pathout)
#   fileout = os.path.join(pathout, 'event_{}.png'.format(event_id))
#   # Compute roc curve points
#   fprs, tprs, thresholds = roc_curve(y_true, y_pred, sample_weight=weights)
#   # Perform all plotting
#   plt.clf()
#   plt.semilogx(fprs, tprs)
#   # Zoom axes
#   plt.xlim([10**-4, 1.0])
#   plt.ylim([0.0, 1.0])
#   # Style
#   plt.xlabel("False Positive Rate (1 - BG Rejection)")
#   plt.ylabel("True Positive Rate (Signal Efficiency)")
#   plt.grid(linestyle=':')
#   # Save
#   plt.savefig(fileout)
#   plt.clf()
