import numpy as np

from .geometry import cart2sphere

pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
pauli_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
pauli_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def pauli_vector():
    """Returns the pauli vector (sigma_x, sigma_y, sigma_z), which is a vector of complex matrices."""
    return np.stack([pauli_x, pauli_y, pauli_z], axis=-1)


def spin_operator(axis):
    """Returns the spin operator along a given axis vector."""
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)

    # project pauli vector
    product = pauli_vector() * axis
    return product.sum(-1)


def spinor_up(axis):
    """Returns the spin up spinor relative to a given axis."""
    r, theta, phi = cart2sphere(axis)

    component_1 = np.cos(theta / 2)
    component_2 = np.exp(1j * phi) * np.sin(theta / 2)

    return np.stack([component_1, component_2], axis=-1)


def spinor_down(axis):
    """Returns the spin down spinor relative to a given axis."""
    r, theta, phi = cart2sphere(axis)

    component_1 = np.sin(theta / 2)
    component_2 = -np.exp(1j * phi) * np.cos(theta / 2)

    return np.stack([component_1, component_2], axis=-1)


def spinor(axis, state):
    """Returns the spin up or spin down spinor relative to a given state depending on the input state -1 or 1."""
    up_state = state == 1
    down_state = state == -1

    up = spinor_up(axis)
    down = spinor_down(axis)

    s = up_state * up + down_state * down
    return s
