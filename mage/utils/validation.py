import numpy as np
import networkx as nx

from collections.abc import Iterable


__all__ = [
    'check_graph_input',
    'check_mask_input',
    'check_random_state'
]


def check_graph_input(G):
    if isinstance(G, np.matrix):
        if G.shape[0] != G.shape[1] or not np.isin(G, [0, 1]).all():
            raise ValueError("""
                             Receiving an numpy matrix, it has to be an
                             adjacency matrix, g.shape = (n, n).
                             """)
        G = nx.Graph(G)
    if isinstance(G, nx.Graph):
        return G
    else:
        raise ValueError('Not supported data type')
    return G


def check_mask_input(mask, size=None):
    if isinstance(mask, np.ndarray):
        if not np.isin(mask, [0, 1]).all():
            raise ValueError("""
                             mask::np.ndarray has to be a binary vector
                             """)
        return mask
    elif isinstance(mask, Iterable):
        if size is None:
            raise ValueError("""
                             If mask is a list of int then you should pass size of the mask.
                             """)
        bin_mask = np.zeros(size)
        bin_mask[np.array(list(mask), dtype=int)] = 1
        return bin_mask
    else:
        raise ValueError('Unknown Mask type')


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.RandomState()
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)
