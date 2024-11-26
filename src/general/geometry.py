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


def cart2sphere(vectors):
    """Returns the spherical coordinates of given cartesian vectors (x, y ,z)."""
    vectors = np.asarray(vectors, dtype=np.float64)

    r = np.linalg.norm(vectors, axis=-1)
    theta = np.arccos(vectors[..., 2] / r)

    xy_vectors = vectors[..., :2]
    r_xy = np.linalg.norm(xy_vectors, axis=-1)[..., np.newaxis]
    r_xy[r_xy == 0] = 1
    xy_unit_vectors = xy_vectors / r_xy

    cos_phi = np.einsum("...i,i", xy_unit_vectors, np.array([1, 0]))
    phi = np.arccos(cos_phi)
    phase_shift = 2 * (vectors[..., 1] < 0) - 1
    phi = (2 * np.pi - phase_shift * phi) % (2 * np.pi)

    return r, theta, phi


def rotate_vectors(vectors, axis, angle):
    """Returns the rotated vectors of a given axis and angle."""
    vectors = np.asarray(vectors, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    angle = np.asarray(angle, dtype=np.float64)[..., np.newaxis]

    axis /= np.sqrt((axis ** 2).sum(-1))[..., np.newaxis]

    rotated_vectors = (
            vectors + np.sin(angle) * np.cross(axis, vectors) +
            (1 - np.cos(angle)) * np.cross(axis, np.cross(axis, vectors))
    )

    return rotated_vectors
