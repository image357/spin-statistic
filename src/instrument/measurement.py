import numpy as np

from src.general import spinor_up, spinor_down
from src.random import random_select


def project_onto_axis(unit_vector, axis):
    """Returns the sign of the projection of the unit_vectors along the measurement axis."""
    unit_vector = np.asarray(unit_vector, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)

    product = unit_vector * axis
    cos_theta = product.sum(-1)
    projection = 2 * np.heaviside(cos_theta, 0) - 1

    return projection.astype(np.int64)


def project_onto_spin_basis(state, axis):
    """Returns the sign of the projection of the spin operator eigen-basis along the measurement axis."""
    # prepare inputs
    state = np.asarray(state, dtype=np.int64)
    if state.ndim != 2:
        state = state[..., np.newaxis]

    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)

    # create (1,0) and (0,1) spin-vectors from state 1 or -1
    up_state = np.ones(shape=(state.shape[0], 2), dtype=np.float64)
    up_state[:, 1] = 0
    down_state = np.ones(shape=(state.shape[0], 2), dtype=np.float64)
    down_state[:, 0] = 0
    spin_state = (state == 1) * up_state + (state == -1) * down_state

    # calculate spin up probability (along given axis)
    axis_up = spinor_up(axis)
    p_up = np.abs(np.einsum("i,...i", axis_up, spin_state)) ** 2

    axis_down = spinor_down(axis)
    p_down = np.abs(np.einsum("i,...i", axis_down, spin_state)) ** 2
    assert np.allclose(p_up + p_down, 1.0), "probabilities of spin projection don't add up to 1.0"

    # sample p_up
    selector_up = random_select(p_up)
    selector_down = np.logical_not(selector_up)

    # project based on selection
    projection = selector_up * 1 + selector_down * (-1)

    return projection.astype(np.int64)
