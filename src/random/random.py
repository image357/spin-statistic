import numpy as np


def uniform_unit_vectors(num):
    """Return num uniformly distributed vectors with unit length.

    See also: https://mathworld.wolfram.com/SpherePointPicking.html
    """
    vectors = np.random.normal(size=(num, 3))
    vectors /= np.sqrt((vectors ** 2).sum(-1))[..., np.newaxis]
    return vectors


def coin_flip(num):
    """Returns the result of num coin flips as either 1 or -1."""
    coins = np.random.randint(0, 2, size=(num, 1))
    return 2 * coins - 1


def random_select(p):
    """Return a truth table based on a random selection with probability p."""
    p = np.array(p, dtype=np.float64)
    vals = np.random.random(size=p.shape)
    selection = np.less(vals, p)
    return selection
