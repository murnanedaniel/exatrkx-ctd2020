import os
import time
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import utils_experiment as utils
from .dataloader import Hit_Pair_Dataset

#####################
#     CONSTANTS     #
#####################
TEST_NAME='Test'

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
  epoch_loss  = 0

  logging.info("Training on {} samples".format(nb_train))
  utils.print_header()
  t0 = time.time()
  elapsed = 0
  for i, (hits_a, hits_b, target) in enumerate(train_loader):
    t1 = time.time()
    hits_a = hits_a.to(DEVICE, non_blocking=True)
    hits_b = hits_b.to(DEVICE, non_blocking=True)
    target = target.to(DEVICE, non_blocking=True)
    '''
    hits_a = hits_a.to(DEVICE)
    hits_b = hits_b.to(DEVICE)
    target = target.to(DEVICE)
    '''
    optimizer.zero_grad()

    emb_h_a = net(hits_a)
    emb_h_b = net(hits_b)

    pred_dist = nn.functional.pairwise_distance(emb_h_a,emb_h_b)
    true_dist = (2*target)-1
    loss = nn.functional.hinge_embedding_loss(pred_dist,true_dist)

    loss.backward()
    optimizer.step()

    score = utils.score_dist_accuracy(pred_dist, target)
    epoch_score += score * 100
    epoch_loss  += loss.item()

    nb_proc = (i+1) * batch_size
    if (((i+0) % (nb_batch//10)) == 0):
        utils.print_eval_stats(nb_proc,
                               epoch_loss/(i+1),
                               epoch_score/(i+1))
    elapsed += time.time()-t1
  logging.info("Model elapsed:  {:.2f}".format(elapsed))
  logging.info("Loader elapsed: {:.2f}".format(time.time()-t0-elapsed))
  logging.info("Total elapsed:  {:.2f}".format(time.time()-t0))
  return epoch_loss / nb_batch, epoch_score / nb_batch


def train(net,
          lr_start,
          batch_size,
          max_nb_epochs,
          experiment_dir,
          train_loader,
          valid_loader):
  optimizer = torch.optim.Adamax(net.parameters(), lr=lr_start)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
  lr_end = lr_start / 10**3
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
        epoch_loss  = 0

        logging.info("\nEvaluating {} {} samples.".format(nb_eval,plot_name))
        utils.print_header()
        for i, (hits_a, hits_b, target) in enumerate(eval_loader):
            hits_a = hits_a.to(DEVICE)
            hits_b = hits_b.to(DEVICE)
            target = target.to(DEVICE)
            t1 = time.time()

            emb_h_a = net(hits_a)
            emb_h_b = net(hits_b)

            pred_dist = nn.functional.pairwise_distance(emb_h_a,emb_h_b)
            true_dist = (2*target)-1
            loss = nn.functional.hinge_embedding_loss(pred_dist,true_dist)

            score = utils.score_dist_accuracy(pred_dist, target)
            epoch_score += score * 100
            epoch_loss += loss.item()

            nb_proc = (i+1) * batch_size
            if ((i+1) % (nb_batch//4)) == 0:
                utils.print_eval_stats(nb_proc,
                                       epoch_loss/(i+1),
                                       epoch_score/(i+1))

    return epoch_loss/nb_batch, epoch_score/nb_batch


def main(args, force=False):

  experiment_dir = os.path.join(args.artifact_storage_path, 'metric_learning_emb')
  
  load_path = os.path.join(args.data_storage_path, 'metric_stage_1')
    
  # Maybe return previously trained model
  best_net_name = os.path.join(experiment_dir, 'best_model.pkl')
  if os.path.isfile(best_net_name) and (not force):
    net = utils.load_model(best_net_name)
    if not force:
      print("Best model loaded from previous run. Not forcing training.")
      return net

  utils.initialize_experiment_if_needed(experiment_dir, evaluate_only=False)
  utils.initialize_logger(experiment_dir)


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

  net = utils.create_or_restore_model(
                                    experiment_dir, 
                                    train_data.get_dim(),
                                    args.nb_hidden, 
                                    args.nb_layer,
                                    args.emb_dim,
                                    stats_path
                                    )
  net.to(DEVICE)
  if next(net.parameters()).is_cuda:
    logging.warning("Working on GPU")
    logging.info("GPU type:\n{}".format(torch.cuda.get_device_name(0)))
  else:
    logging.warning("Working on CPU")
    

  train(net,
        args.lr_start,
        args.batch_size,
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
  test_stats = evaluate(net, experiment_dir, args.batch_size, test_dataloader, TEST_NAME)
  utils.save_test_stats(experiment_dir, test_stats)
  logging.info("Test score:  {:3.2f}".format(test_stats[1]))

  return net

if __name__ == "__main__":

    args = read_args()
    main(args)
