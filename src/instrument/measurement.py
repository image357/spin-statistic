from .helper import spin_operator
from src.random import random_select
import numpy as np


def project_onto_axis(unit_vectors, axis_vector):
    """Returns the sign of the projection of the unit_vectors along the measurement axis_vector."""
    axis_vector = np.array(axis_vector, dtype=np.float64)
    axis_vector /= np.linalg.norm(axis_vector)
    product = unit_vectors * axis_vector
    cos_theta = product.sum(-1)
    projection = 2 * np.heaviside(cos_theta, 0) - 1
    return projection.astype(np.int64)


def project_onto_spin_basis(state, axis_vector):
    """Returns the sign of the projection of the spin operator eigen-basis along the measurement axis_vector."""
    # prepare inputs
    state = np.array(state, dtype=np.float64)
    if state.ndim != 2:
        state = state[..., np.newaxis]

    axis_vector = np.array(axis_vector, dtype=np.float64)
    axis_vector /= np.linalg.norm(axis_vector)

    # create (1,0) and (0,1) spin-vectors from state 1 or -1
    up_state = np.ones(shape=(state.shape[0], 2), dtype=np.float64)
    up_state[:, 1] = 0
    down_state = np.ones(shape=(state.shape[0], 2), dtype=np.float64)
    down_state[:, 0] = 0
    spin_state = (state == 1) * up_state + (state == -1) * down_state

    # construct projection basis
    so = spin_operator(axis_vector)
    eigvals, eigvecs = np.linalg.eig(so)

    # calculate spin projections
    p0 = np.abs(np.einsum("i,...i", eigvecs[:, 0].conj(), spin_state))**2
    p1 = np.abs(np.einsum("i,...i", eigvecs[:, 1].conj(), spin_state))**2

    # sample p0
    selector0 = random_select(p0)
    selector1 = np.logical_not(selector0)

    # project based on selection
    projection = selector0 * np.real(eigvals[0]) + selector1 * np.real(eigvals[1])
    projection = 2 * np.heaviside(projection, 0) - 1

    return projection.astype(np.int64)
