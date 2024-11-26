import numpy as np


def sphere2cart(r, theta, phi):
    """Returns a measurement axis expressed in spherical coordinates phi and theta."""
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    r = np.asarray(r)

    axis_vector_x = r * sin_theta * cos_phi
    axis_vector_y = r * sin_theta * sin_phi
    axis_vector_z = r * cos_theta * np.ones_like(sin_phi)

    axis_vector = np.stack([axis_vector_x, axis_vector_y, axis_vector_z], axis=-1)
    return axis_vector
