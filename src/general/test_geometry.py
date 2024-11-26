from unittest import TestCase

from .geometry import sphere2cart, cart2sphere

import numpy as np


class Test(TestCase):
    def test_sphere2cart(self):
        vectors = sphere2cart(5, [0, np.pi / 2, np.pi], [0, np.pi / 2, np.pi])
        self.assertTrue(np.allclose(vectors[0, :], [0, 0, 5]))
        self.assertTrue(np.allclose(vectors[1, :], [0, 5, 0]))
        self.assertTrue(np.allclose(vectors[2, :], [0, 0, -5]))

    def test_cart2sphere(self):
        r = [1, 1, 7, 8, 9]
        theta = [0, np.pi, 1, 2, 3]
        phi = [np.pi / 2, np.pi/2, 1, 3, 6]

        vectors = sphere2cart(r, theta, phi)
        new_r, new_theta, new_phi = cart2sphere(vectors)

        self.assertTrue(np.allclose(new_r, r))
        self.assertTrue(np.allclose(new_theta, theta))
        self.assertTrue(np.allclose(new_phi, phi))
