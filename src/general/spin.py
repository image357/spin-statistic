import numpy as np

pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
pauli_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
pauli_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def pauli_vector():
    """Returns the pauli vector (sigma_x, sigma_y, sigma_z), which is a vector of complex matrices."""
    return np.stack([pauli_x, pauli_y, pauli_z], axis=-1)


def spin_operator(axis_vector):
    """Returns the spin operator along a given axis vector."""
    axis_vector = np.array(axis_vector, dtype=np.float64)
    axis_vector /= np.linalg.norm(axis_vector)

    # project pauli vector
    product = pauli_vector() * axis_vector
    return product.sum(-1)

def spinor(axis_vector, state):
    return -1
