"""
Python module for holding our PyTorch models.
"""

def get_model(name, **model_args):
    """
    Top-level factory function for getting your models.
    """
    if name == 'agnn_original':
        from .agnn_original import GNNSegmentClassifier
        return GNNSegmentClassifier(**model_args)
    elif name == 'agnn':
        from .agnn import GNNSegmentClassifier
        return GNNSegmentClassifier(**model_args)
    elif name == 'mpnn':
        from .mpnn import GNN
        return GNN(**model_args)
    elif name == 'resgnn':
        from .resgnn import GNNSegmentClassifier
        return GNNSegmentClassifier(**model_args)
    elif name == 'tripgnn':
        from .tripgnn import GNNSegmentClassifier
        return GNNSegmentClassifier(**model_args)
    else:
        raise Exception('Model %s unknown' % name)
