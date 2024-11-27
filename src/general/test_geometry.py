from unittest import TestCase

from .geometry import sphere2cart, cart2sphere, rotate_vector

import numpy as np


class Test(TestCase):
    def test_sphere2cart(self):
        vectors = sphere2cart(5, [0, np.pi / 2, np.pi], [0, np.pi / 2, np.pi])
        self.assertTrue(np.allclose(vectors[0, :], [0, 0, 5]))
        self.assertTrue(np.allclose(vectors[1, :], [0, 5, 0]))
        self.assertTrue(np.allclose(vectors[2, :], [0, 0, -5]))

        vectors = sphere2cart(1, 2, 3)
        self.assertEqual((3,), vectors.shape)

    def test_cart2sphere(self):
        r = [1, 1, 7, 8, 9]
        theta = [0, np.pi, 1, 2, 3]
        phi = [np.pi / 2, np.pi / 2, 1, 3, 6]

        vectors = sphere2cart(r, theta, phi)
        new_r, new_theta, new_phi = cart2sphere(vectors)

        self.assertTrue(np.allclose(new_r, r))
        self.assertTrue(np.allclose(new_theta, theta))
        self.assertTrue(np.allclose(new_phi, phi))

        new_r, new_theta, new_phi = cart2sphere([1, 0, 0])
        self.assertAlmostEqual(1, new_r)
        self.assertAlmostEqual(np.pi / 2, new_theta)
        self.assertAlmostEqual(0, new_phi)

    def test_rotate_vectors(self):
        vectors = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 1],
                [1, 2, 3],
                [1, 2, -1],
                [3, -2, 1],
            ],
            dtype=np.float64
        )

        r, theta, phi = cart2sphere(vectors)
        axis = sphere2cart(r, theta + np.pi / 2, phi)
        angle = np.linspace(0, np.pi, vectors.shape[0])

        rotated_vectors = rotate_vector(vectors, axis, angle)

        vectors /= np.sqrt((vectors ** 2).sum(-1))[..., np.newaxis]
        rotated_vectors /= np.sqrt((rotated_vectors ** 2).sum(-1))[..., np.newaxis]

        cos_alpha = np.einsum("...i,...i", rotated_vectors, vectors)
        alpha = np.arccos(cos_alpha)
        self.assertTrue(np.allclose(angle, alpha))

        rotated_vector = rotate_vector([0, 0, 1], [1, 0, 0], np.pi / 2)
        self.assertTrue(np.allclose([0, -1, 0], rotated_vector))
