import numpy as np


def sphere2cart(r, theta, phi):
    """Returns a cartesian vector expressed in spherical coordinates r, theta and phi."""
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    r = np.asarray(r, dtype=np.float64)

    axis_vector_x = r * sin_theta * cos_phi
    axis_vector_y = r * sin_theta * sin_phi
    axis_vector_z = r * cos_theta * np.ones_like(sin_phi)

    axis_vector = np.stack([axis_vector_x, axis_vector_y, axis_vector_z], axis=-1)
    return axis_vector


def cart2sphere(vector):
    """Returns the spherical coordinates of given cartesian vectors (x, y ,z)."""
    vector = np.asarray(vector, dtype=np.float64)

    r = np.linalg.norm(vector, axis=-1)
    theta = np.arccos(vector[..., 2] / r)

    xy_vector = vector[..., :2]
    r_xy = np.linalg.norm(xy_vector, axis=-1)[..., np.newaxis]
    r_xy[r_xy == 0] = 1
    xy_unit_vector = xy_vector / r_xy

    cos_phi = np.einsum("...i,i", xy_unit_vector, np.array([1, 0], dtype=np.float64))
    phi = np.arccos(cos_phi)
    phase_shift = 2 * (vector[..., 1] < 0) - 1
    phi = (2 * np.pi - phase_shift * phi) % (2 * np.pi)

    return r, theta, phi


def rotate_vector(vector, axis, angle):
    """Returns the rotated vector of a given rotation-axis and angle."""
    vector = np.asarray(vector, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    angle = np.asarray(angle, dtype=np.float64)[..., np.newaxis]

    axis /= np.sqrt((axis ** 2).sum(-1))[..., np.newaxis]

    rotated_vector = (
            vector + np.sin(angle) * np.cross(axis, vector) +
            (1 - np.cos(angle)) * np.cross(axis, np.cross(axis, vector))
    )

    return rotated_vector
