import numpy as np
import itertools
import networkx as nx

from more_itertools import powerset
from functools import partial
from scipy.sparse.csgraph import connected_components

from mage.utils.validation import check_random_state


__all__ = [
    'shapley_taylor_indices',
    'myerson_interaction_indices',
]

"""
SHAPLEY_TAYLOR_INDICES
"""


def delta_fn(S, T, fn):
    s = len(S)
    T_set = set(T)
    ret = 0

    for W in powerset(S):
        w = len(W)
        value = fn(frozenset(T_set.union(W)))
        ret += (-1)**(w - s) * value

    return ret


def shapley_taylor_indices(num_players, fn, ord=2, num_samples=500, random_state=None, return_indices=True):
    rng = check_random_state(random_state)
    indices = np.zeros([num_players]*ord, dtype=np.float64)
    sum_inds = np.zeros_like(indices)
    cnt_inds = np.zeros_like(indices)
    players = np.array(list(range(num_players)))
    # print(players)

    for _ in range(num_samples):
        p = np.array(rng.permutation(num_players))
        inv_p = np.zeros_like(p)
        inv_p[p] = players

        for S in itertools.combinations(players, ord):
            i_k = np.min(inv_p[np.array(S)])
            T = p[:i_k]

            delta = delta_fn(S, T, fn)

            if return_indices:
                for p_S in itertools.permutations(S):
                    sum_inds[p_S] += delta
                    cnt_inds[p_S] += 1

    if return_indices:
        indices = np.divide(sum_inds, cnt_inds, out=np.zeros_like(sum_inds), where=(cnt_inds != 0))

    for r in range(1, ord):
        for S in itertools.combinations(players, r):
            delta = delta_fn(S, tuple(), fn)
            if return_indices:
                # Temporary solution, a not good practice
                # We access the index of a subset S size h by indices[T]
                # where T = (S, S[0], S[0], ...S[0]), |S| = h, |T| = ord
                for i in range(len(S)):
                    p_S = S + (S[i],) * (ord - len(S))
                    indices[p_S] = delta

    return indices


def shapley_interaction_indices(num_players, fn, ord=2, num_samples=500,
                                random_state=None, return_indices=True):
    if ord > 2:
        raise NotImplementedError

    rng = check_random_state(random_state)
    indices = np.zeros([num_players]*ord, dtype=np.float32)
    sum_inds = np.zeros_like(indices)
    cnt_inds = np.zeros_like(indices)

    for _ in range(num_samples):
        p = np.array(rng.permutation(num_players))

        for l in range(1, ord+1):
            for k in range(0, num_players - l):
                T = p[:k]
                S = p[k: k+l]

                delta = delta_fn(S, T, fn)

                if return_indices:
                    for p_S in itertools.permutations(S):
                        if len(S) < ord:
                            p_S = (S[0], S[0])
                        sum_inds[p_S] += delta
                        cnt_inds[p_S] += 1

    if return_indices:
        indices = np.divide(sum_inds, cnt_inds, out=np.zeros_like(sum_inds), where=(cnt_inds != 0))

    return indices
