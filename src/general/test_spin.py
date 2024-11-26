from unittest import TestCase

from .spin import pauli_x, pauli_y, pauli_z
from .spin import pauli_vector, spin_operator

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
