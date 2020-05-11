"""Helper code for using the Cray DL Plugin"""

import dl_comm.torch as cdl

def init_workers_cray():
    rank = cdl.get_rank()
    n_ranks = cdl.get_nranks()
    return rank, n_ranks

def distribute_optimizer(optimizer, n_teams=1, n_threads=2):
    """
    Wrap the optimizer in order to use the Plugin's communication.

    Arguments:
        n_teams: Number of teams you'll be training
        n_threads: Number of communication threads per team
    """
    optimizer = cdl.DistributedOptimizer(optimizer, nteam=n_teams,
                                         nthread_per_team=n_threads)
    return optimizer
