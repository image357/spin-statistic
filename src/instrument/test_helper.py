from unittest import TestCase
from src.instrument import measurement_axis, pauli_vector, spin_operator
from src.instrument.helper import pauli_x, pauli_y, pauli_z
import numpy as np


class Test(TestCase):
    def test_measurement_axis(self):
        ma = measurement_axis([0, np.pi / 2, np.pi], [0, np.pi / 2, np.pi])
        self.assertTrue(np.allclose(ma[0, :], [0, 0, 1]))
        self.assertTrue(np.allclose(ma[1, :], [0, 1, 0]))
        self.assertTrue(np.allclose(ma[2, :], [0, 0, -1]))

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
