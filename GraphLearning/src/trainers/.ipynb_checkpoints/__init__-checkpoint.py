"""
Python module for holding our PyTorch trainers.

Trainers here inherit from the BaseTrainer and implement the logic for
constructing the model as well as training and evaluation.
"""

def get_trainer(name, **trainer_args):
    """
    Factory function for retrieving a trainer.
    """
    if name == 'gnn_dense':
        from .gnn_dense import DenseGNNTrainer
        return DenseGNNTrainer(**trainer_args)
    elif name == 'gnn_sparse':
        from .gnn_sparse import SparseGNNTrainer
        return SparseGNNTrainer(**trainer_args)
    else:
        raise Exception('Trainer %s unknown' % name)
