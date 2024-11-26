from unittest import TestCase

from .random import uniform_unit_vectors, coin_flip, random_select

import numpy as np


class Test(TestCase):
    def test_uniform_unit_vectors_normalization(self):
        uv = uniform_unit_vectors(100)
        for i in range(100):
            self.assertAlmostEqual(uv[i, 0] ** 2 + uv[i, 1] ** 2 + uv[i, 2] ** 2, 1)

    def test_coin_flip(self):
        coins = coin_flip(100)
        total = np.sum(coins == 1) + np.sum(coins == -1)
        self.assertEqual(total, 100)

    def test_random_select(self):
        selection = random_select([[0.5], [1]])
        self.assertEqual(selection.shape, (2, 1))
