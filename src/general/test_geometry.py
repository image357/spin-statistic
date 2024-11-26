from unittest import TestCase

from .geometry import sphere2cart

import numpy as np


class Test(TestCase):
    def test_sphere2cart(self):
        ma = sphere2cart(5, [0, np.pi / 2, np.pi], [0, np.pi / 2, np.pi])
        self.assertTrue(np.allclose(ma[0, :], [0, 0, 5]))
        self.assertTrue(np.allclose(ma[1, :], [0, 5, 0]))
        self.assertTrue(np.allclose(ma[2, :], [0, 0, -5]))
