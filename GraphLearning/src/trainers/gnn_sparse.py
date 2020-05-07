"""
This module defines a generic trainer for simple models and datasets.
"""

# System
import logging

# Externals
import torch

# Locals
from .gnn_base import GNNBaseTrainer
from ..utils.checks import get_weight_norm, get_grad_norm

class SparseGNNTrainer(GNNBaseTrainer):
    """Trainer code for sparse GNN."""

    def train_epoch(self, data_loader):
        """Train for one epoch"""
        self.model.train()

        # Prepare summary information
        summary = dict()
        sum_loss = 0

        # Loop over training batches
        for i, batch in enumerate(data_loader):
            batch = batch.to(self.device)
            self.model.zero_grad()
            batch_output = self.model(batch)
            batch_loss = self.loss_func(batch_output, batch.y, weight=batch.w)
            batch_loss.backward()
            self.optimizer.step()
            sum_loss += batch_loss.item()

            # Dump additional debugging information
            if self.logger.isEnabledFor(logging.DEBUG):
                l1 = get_weight_norm(self.model, 1)
                l2 = get_weight_norm(self.model, 2)
                grad_norm = get_grad_norm(self.model)
                self.logger.debug('  train batch %i loss %.4f l1 %.2f l2 %.4f grad %.3f idx %i',
                                  i, batch_loss.item(), l1, l2, grad_norm, batch.i[0].item())

        # Summarize the epoch
        n_batches = i + 1
        summary['lr'] = self.optimizer.param_groups[0]['lr']
        summary['train_loss'] = sum_loss / n_batches
        summary['l1'] = get_weight_norm(self.model, 1)
        summary['l2'] = get_weight_norm(self.model, 2)
        self.logger.debug(' Processed %i batches', n_batches)
        self.logger.debug(' Model LR %f l1 %.2f l2 %.2f',
                          summary['lr'], summary['l1'], summary['l2'])
        self.logger.info('  Training loss: %.3f', summary['train_loss'])
        return summary

    @torch.no_grad()
    def evaluate(self, data_loader):
        """"Evaluate the model"""
        self.model.eval()

        # Prepare summary information
        summary = dict()
        sum_loss = 0
#         sum_correct = 0
#         sum_total = 0
        
        true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0

        # Loop over batches
        for i, batch in enumerate(data_loader):
            batch = batch.to(self.device)

            # Make predictions on this batch
            batch_output = self.model(batch)
            batch_loss = self.loss_func(batch_output, batch.y).item()
            sum_loss += batch_loss
            
            # Counts for accuracy, efficiency & purity
            batch_pred = torch.sigmoid(batch_output)
            pred_true = batch_pred > 0.5
            ground_true = batch.y > 0.5
            
            true_pos += (pred_true & ground_true).sum().item()
            false_pos += (pred_true & ~ground_true).sum().item()
            true_neg += (~pred_true & ~ground_true).sum().item()
            false_neg += (~pred_true & ground_true).sum().item()
            
            self.logger.debug(' valid batch %i, loss %.4f', i, batch_loss)
        
        # Summarize the validation epoch
        n_batches = i + 1        
        summary['valid_loss'] = sum_loss / n_batches
        summary['valid_acc'] = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        self.logger.debug(' Processed %i samples in %i batches',
                          len(data_loader.sampler), n_batches)
        
        # This handles that distributed mode doesn't have access to true efficiency/purity
        if self.distributed_mode in ['ddp-file', 'ddp-mpi']: 
            self.logger.info('  Validation loss: %.3f acc: %.3f' %
                             (summary['valid_loss'],  summary['valid_acc']))          

        else:
            summary['valid_eff'] = true_pos / (true_pos + false_neg)
            summary['valid_pur'] = true_pos / (true_pos + false_pos)
            self.logger.info('  Validation loss: %.3f acc: %.3f pur: %.3f eff: %.3f' %
                             (summary['valid_loss'],  summary['valid_acc'], summary['valid_pur'], summary['valid_eff']))
        return summary

    @torch.no_grad()
    def predict(self, data_loader):
        preds, targets = [], []
        for batch in data_loader:
            preds.append(torch.sigmoid(self.model(batch)).squeeze(0))
            targets.append(batch.y.squeeze(0))
        return preds, targets
    
    @torch.no_grad()
    def device_predict(self, data_loader):
        preds, targets = [], []
        for i, batch in enumerate(data_loader):
            data = batch.to(self.device)
            if i%100==0: logging.info("Batch %i" % i)
            preds.append(torch.sigmoid(self.model(data).cpu()).squeeze(0))
            targets.append(data.y.squeeze(0).cpu())
        return preds, targets

def _test():
    t = SparseGNNTrainer(output_dir='./')
    t.build_model()
