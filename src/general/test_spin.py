from unittest import TestCase

from .spin import pauli_x, pauli_y, pauli_z
from .spin import pauli_vector, spin_operator
from .spin import spinor_up, spinor_down, spinor

import numpy as np


class Test(TestCase):
    def test_pauli_vector(self):
        pv = pauli_vector()
        self.assertEqual(pv.shape, (2, 2, 3))

    def test_spin_operator(self):
        so = spin_operator([1, 0, 0])
        self.assertEqual(so.shape, (2, 2))
        self.assertTrue(np.allclose(so, pauli_x))

        so = spin_operator([0, 1, 0])
        self.assertEqual(so.shape, (2, 2))
        self.assertTrue(np.allclose(so, pauli_y))

        so = spin_operator([0, 0, 1])
        self.assertEqual(so.shape, (2, 2))
        self.assertTrue(np.allclose(so, pauli_z))

    def test_spinor_up_down(self):
        axis = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ])

        up = spinor_up(axis)
        down = spinor_down(axis)

        up_up = np.einsum("...i,...i", up.conj(), up)
        down_down = np.einsum("...i,...i", down.conj(), down)
        up_down = np.einsum("...i,...i", up.conj(), down)

        for i in range(len(axis)):
            self.assertAlmostEqual(1.0 + 0.0j, up_up[i])
            self.assertAlmostEqual(1.0 + 0.0j, down_down[i])
            self.assertAlmostEqual(0.0 + 0.0j, up_down[i])

        for i in range(len(axis)):
            so = spin_operator(axis[i])
            so_up = np.einsum("ij,j", so, up[i])
            so_down = np.einsum("ij,j", so, down[i])

            self.assertTrue(np.allclose(up[i], 1 * so_up))
            self.assertTrue(np.allclose(down[i], -1 * so_down))

    def test_spinor(self):
        axis = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ])
        state = [1, -1, 1, -1]

        s = spinor(axis, state)
        self.assertEqual((4, 2), s.shape)

        s = spinor([0, 0, 1], 1)
        self.assertAlmostEqual(s[0], 1.0 + 0.0j)
        self.assertAlmostEqual(s[1], 0.0 + 0.0j)

        s = spinor([0, 0, 1], -1)
        self.assertAlmostEqual(s[0], 0.0 + 0.0j)
        self.assertAlmostEqual(s[1], 0.0 - 1.0j)

        s = spinor([0, 0, -1], 1)
        self.assertAlmostEqual(s[0], 0.0 + 0.0j)
        self.assertAlmostEqual(s[1], 0.0 + 1.0j)

        s = spinor([0, 0, -1], -1)
        self.assertAlmostEqual(s[0], 1.0 + 0.0j)
        self.assertAlmostEqual(s[1], 0.0 + 0.0j)
