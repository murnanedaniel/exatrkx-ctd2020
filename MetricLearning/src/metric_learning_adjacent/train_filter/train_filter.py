import os
import time
import logging
import numpy as np

from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import mlp_model
from . import utils_experiment as utils
from .dataloader import Hit_Pair_Dataset

#################################################
#                   CONSTANTS                   #
#################################################
NB_SAMPLES_FOR_NORM=10**5

#####################################
#               DEVICE              #
#####################################
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

#######################
#     EXPERIMENT      #
#######################

def train_one_epoch(net, batch_size, optimizer, train_loader):
  net.train()

  nb_batch = len(train_loader)
  nb_train = nb_batch * batch_size
  epoch_score = 0
  epoch_auc   = 0
  epoch_loss  = 0

  logging.info("Training on {} samples".format(nb_train))
  utils.print_header()
  t0 = time.time()
  elapsed = 0
  for i, (hits, target) in enumerate(train_loader):
    t1 = time.time()
    hits = hits.to(DEVICE, non_blocking=True)
    target = target.to(DEVICE, non_blocking=True)
    optimizer.zero_grad()

    pred = net(hits)

    loss = nn.functional.binary_cross_entropy(pred, target)

    loss.backward()
    optimizer.step()

    score = utils.score_dist_accuracy(pred, target)
    auc   = roc_auc_score(target.data.cpu(), pred.data.cpu())
    epoch_score += score * 100
    epoch_auc   += auc
    epoch_loss  += loss.item()

    nb_proc = (i+1) * batch_size
    if (((i+0) % (nb_batch//10)) == 0):
        print("  {:8d}:  Loss {:7.3f}  Acc {:5.2f}  AUC {:4.3f}".format(
                               nb_proc,
                               epoch_loss/(i+1),
                               epoch_score/(i+1),
                               epoch_auc/(i+1)))
    elapsed += time.time()-t1
  logging.info("Model elapsed:  {:.2f}".format(elapsed))
  logging.info("Loader elapsed: {:.2f}".format(time.time()-t0-elapsed))
  logging.info("Total elapsed:  {:.2f}".format(time.time()-t0))
  return epoch_loss / nb_batch, epoch_score / nb_batch


def train(net,
          batch_size,
          lr_start,
          max_nb_epochs,
          experiment_dir,
          train_loader,
          valid_loader):
  optimizer = torch.optim.Adamax(net.parameters(), lr=lr_start)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
  lr_end = lr_start / 100
  best_loss = 10**10
  # Nb epochs completed tracked in case training interrupted
  for i in range(max_nb_epochs):
    t0 = time.time()
    logging.info("\nEpoch {}".format(i+1))
    logging.info("Learning rate: {0:.3g}".format(lr_start))

    train_stats = train_one_epoch(net, batch_size, optimizer, train_loader)
    val_stats = evaluate(net, experiment_dir, batch_size, valid_loader, 'Valid')
                                
    logging.info("Train accuracy: {:3.2f}".format(train_stats[1]))
    logging.info("Valid accuracy: {:3.2f}".format(val_stats[1]))
    utils.track_epoch_stats(i, 
                            lr_start, 
                            train_stats, 
                            val_stats,
                            experiment_dir)

    scheduler.step(val_stats[0])
    lr_start = optimizer.param_groups[0]['lr']

    if (val_stats[0] < best_loss):
      logging.warning("Best performance on valid set.")
      best_loss = val_stats[0]
      # utils.update_best_plots(experiment_dir)
      utils.save_best_model(experiment_dir, net)

    utils.save_epoch_model(experiment_dir, net)

    logging.info("Epoch took {} seconds.".format(int(time.time()-t0)))

    if lr_start < lr_end:
        break
  logging.warning("Training completed.")


def evaluate(net, experiment_dir, batch_size, eval_loader, plot_name):
    nb_batch = len(eval_loader)
    nb_eval = nb_batch * batch_size
    net.eval()

    with torch.autograd.no_grad():
        epoch_score = 0
        epoch_auc   = 0
        epoch_loss  = 0

        logging.info("\nEvaluating {} {} samples.".format(nb_eval,plot_name))
        utils.print_header()
        for i, (hits, target) in enumerate(eval_loader):
            hits = hits.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)

            pred = net(hits)
            loss = nn.functional.binary_cross_entropy(pred, target)

            score = utils.score_dist_accuracy(pred, target)
            auc   = roc_auc_score(target.data.cpu(), pred.data.cpu())
            epoch_score += score * 100
            epoch_auc  += auc
            epoch_loss += loss.item()

            nb_proc = (i+1) * batch_size
            if ((i+1) % (nb_batch//4)) == 0:
                print("  {:8d}:  Loss {:7.3f}  Acc {:5.2f}  AUC {:4.3f}".format(
                                       nb_proc,
                                       epoch_loss/(i+1),
                                       epoch_score/(i+1),
                                       epoch_auc/(i+1)))

    return epoch_loss/nb_batch, epoch_score/nb_batch

def set_model_norm(loader, net):
    nb_samples = len(loader.dataset)
    nb_iter = min(NB_SAMPLES_FOR_NORM, nb_samples)
    batch_size = nb_samples//len(loader)
    nb_batch_norm = 1+nb_iter//batch_size

    all_feats = []
    net.eval()
    with torch.no_grad():
        for i, (hits, truth) in enumerate(loader):
            hits.to(DEVICE, non_blocking=True)
            hits_aug = mlp_model.augment_features(hits)
            all_feats.append(hits_aug)
            if (i >= nb_batch_norm):
                break

        all_feats = torch.cat(all_feats, dim=0)
        mean = torch.mean(all_feats, dim=0)
        std  = torch.std(all_feats, dim=0)
        net.set_norm(mean, std)


#############################################
#                   MAIN                    #
#############################################

def main(args, force=False):

  experiment_dir = os.path.join(args.artifact_storage_path, 'metric_learning_filter')

  load_path = os.path.join(args.data_storage_path, 'metric_stage_2')
  
  # Maybe return previously trained model
  best_net_name = os.path.join(experiment_dir, 'best_model.pkl')
  if os.path.isfile(best_net_name) and (not force):
    net = utils.load_model(best_net_name)
    if not force:
      print("Best filter model loaded from previous run. Not forcing training.")
      return net

  utils.initialize_experiment_if_needed(experiment_dir, evaluate_only=False)
#   utils.initialize_logger(experiment_dir)


  train_path = os.path.join(load_path, 'train.pickle')
  valid_path = os.path.join(load_path, 'valid.pickle')
  test_path  = os.path.join(load_path, 'test.pickle')
  stats_path = os.path.join(load_path, 'stats.yml')

  train_data = Hit_Pair_Dataset(train_path, 10**8)
  valid_data = Hit_Pair_Dataset(valid_path, 10**8)
  test_data  = Hit_Pair_Dataset(test_path, 10**8)

  train_dataloader = DataLoader(train_data,
                                batch_size=args.batch_size,
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True,
                                num_workers=8)
  valid_dataloader = DataLoader(valid_data,
                                batch_size=args.batch_size,
                                drop_last=True,
                                pin_memory=True,
                                num_workers=8)
  test_dataloader  = DataLoader(test_data,
                                batch_size=args.batch_size,
                                drop_last=True,
                                pin_memory=True,
                                num_workers=8)

  input_dim = train_data.get_input_dim()
  logging.info("Input dimension: {}".format(input_dim))

  net = utils.create_or_restore_model(
                                    experiment_dir, 
                                    input_dim,
                                    args.nb_hidden, 
                                    args.nb_layer,
                                    args.emb_dim
                                    )
  net.to(DEVICE)
  if next(net.parameters()).is_cuda:
    logging.warning("Working on GPU")
    logging.info("GPU type:\n{}".format(torch.cuda.get_device_name(0)))
  else:
    logging.warning("Working on CPU")

  # Set model normalization if needed
  if (not net.norm_set):
    logging.info("Setting normalization")
    set_model_norm(train_dataloader, net)

  train(net,
        args.batch_size,
        args.lr_start,
        args.max_nb_epochs,
        experiment_dir,
        train_dataloader,
        valid_dataloader)

  # Perform evaluation over test set
  try:
    net = utils.load_best_model(experiment_dir).to(DEVICE)
    logging.warning("\nBest model loaded for evaluation on test set.")
  except:
    logging.warning("\nCould not load best model for test set. Using current.")
  test_stats = evaluate(net, experiment_dir, args.batch_size, test_dataloader, "test")
  utils.save_test_stats(experiment_dir, test_stats)
  logging.info("Test score:  {:3.2f}".format(test_stats[1]))

  return net

if __name__ == "__main__":
    
    args = read_args()
    main()
