from itertools import islice
import numpy as np


def vpack(arrays, shape, fill, dtype= None):
    """like `np.vstack` but for `arrays` of different lengths in the first
    axis.  shorter ones will be padded with `fill` at the end.

    """
    array = np.full(shape, fill, dtype)
    for row, arr in zip(array, arrays):
        row[:len(arr)] = arr
    return array


def partition(n, m, discard= False):
    """yields pairs of indices which partitions `n` nats by `m`.  if not
    `discard`, also yields the final incomplete partition.

    """
    steps = range(0, 1 + n, m)
    yield from zip(steps, steps[1:])
    if n % m and not discard:
        yield n - (n % m), n


def sample(n, seed= 0):
    """yields samples from `n` nats."""
    data = list(range(n))
    while True:
        np.random.seed(seed)
        np.random.shuffle(data)
        yield from data


def unison_shfl(l1, l2, seed=0):
    l1_shfl, l2_shfl = [], []
    shfl_idx = np.arange(len(l1))
    np.random.seed(seed)
    np.random.shuffle(shfl_idx)
    for i in shfl_idx:
        l1_shfl.append(l1[i])
        l2_shfl.append(l2[i])
    return l1_shfl, l2_shfl
