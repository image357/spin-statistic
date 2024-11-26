from unittest import TestCase

from .measurement import project_onto_axis, project_onto_spin_basis

import numpy as np


class Test(TestCase):
    def test_project_onto_axis(self):
        unit_vectors = np.array([
            [1, 0, 0],
            [-1, 0, 0],

            [0, 1, 0],
            [0, -1, 0],

            [0, 0, 1],
            [0, 0, -1],

            [1 / np.sqrt(2), 0, 1 / np.sqrt(2)],
            [-1 / np.sqrt(2), 0, 1 / np.sqrt(2)],
            [-1 / np.sqrt(2), 0, -1 / np.sqrt(2)],
            [1 / np.sqrt(2), 0, -1 / np.sqrt(2)],

            [1 / np.sqrt(2), 1 / np.sqrt(2), 0],
            [-1 / np.sqrt(2), 1 / np.sqrt(2), 0],
            [-1 / np.sqrt(2), -1 / np.sqrt(2), 0],
            [1 / np.sqrt(2), -1 / np.sqrt(2), 0],

            [0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
            [0, -1 / np.sqrt(2), 1 / np.sqrt(2)],
            [0, -1 / np.sqrt(2), -1 / np.sqrt(2)],
            [0, 1 / np.sqrt(2), -1 / np.sqrt(2)],
        ])

        projection = project_onto_axis(unit_vectors, [1, 0, 0])

        expected = [
            1,
            -1,

            -1,
            -1,

            -1,
            -1,

            1,
            -1,
            -1,
            1,

            1,
            -1,
            -1,
            1,

            -1,
            -1,
            -1,
            -1,
        ]

        for i in range(projection.shape[0]):
            self.assertEqual(projection[i], expected[i])

    def test_project_from_pauli_vector(self):
        projection = project_onto_spin_basis([1, 1, -1], [0, 0, 1])
        expected = [1, 1, -1]
        for i in range(projection.shape[0]):
            self.assertEqual(projection[i], expected[i])

        projection = project_onto_spin_basis([1, 1, -1], [0, 0, -1])
        expected = [-1, -1, 1]
        for i in range(projection.shape[0]):
            self.assertEqual(projection[i], expected[i])
