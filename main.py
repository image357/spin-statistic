import matplotlib.pyplot as plt
import numpy as np

from src.general import sphere2cart, cart2sphere, rotate_vector, spinor
from src.instrument import project_onto_spin_basis
from src.random import coin_flip

# statistics
n_angles = 30
n_samples_per_angle = 10000

# left measurement axis
measurement_axis_left = sphere2cart(1, 0, 0)

# construct rotation axis
r, theta, phi = cart2sphere(measurement_axis_left)
theta = theta + np.pi / 2
rotation_axis = sphere2cart(1, theta, phi)
alpha = np.linspace(0, np.pi, n_angles)

# construct right measurement axis from rotation
measurement_axis_right = rotate_vector(measurement_axis_left, rotation_axis, alpha)

# check angles
cos_alpha = (measurement_axis_right * measurement_axis_left).sum(-1)
assert np.allclose(alpha, np.arccos(cos_alpha))

# this is an implicit requirement
source_axis = measurement_axis_left

# prepare result buffers
p_left_up = np.zeros(n_angles)
p_left_down = np.zeros(n_angles)
p_right_up = np.zeros(n_angles)
p_right_down = np.zeros(n_angles)

p_up_up = np.zeros(n_angles)
p_up_down = np.zeros(n_angles)
p_down_up = np.zeros(n_angles)
p_down_down = np.zeros(n_angles)

# run experiments
for i in range(n_angles):
    # prepare counters for each angle
    sum_left_up = 0
    sum_left_down = 0
    sum_right_up = 0
    sum_right_down = 0

    sum_up_up = 0
    sum_up_down = 0
    sum_down_up = 0
    sum_down_down = 0

    # sample left "hidden" variables
    left = coin_flip(n_samples_per_angle)

    # (anti-)correlate right and left measurements
    right = -left

    # get spinor
    left_spinor = spinor(source_axis, left)
    right_spinor = spinor(source_axis, right)

    # perform measurement projections
    pj_left = project_onto_spin_basis(left_spinor, measurement_axis_left)
    pj_right = project_onto_spin_basis(right_spinor, measurement_axis_right[i])

    # count left and right measurements
    sum_left_up += np.sum(pj_left == 1)
    sum_left_down += np.sum(pj_left == -1)
    sum_right_up += np.sum(pj_right == 1)
    sum_right_down += np.sum(pj_right == -1)

    # count correlated measurement occurrence
    sum_up_up += np.logical_and(pj_left == 1, pj_right == 1).sum()
    sum_up_down += np.logical_and(pj_left == 1, pj_right == -1).sum()
    sum_down_up += np.logical_and(pj_left == -1, pj_right == 1).sum()
    sum_down_down += np.logical_and(pj_left == -1, pj_right == -1).sum()

    # calculate left and right probabilities
    p_left_up[i] = sum_left_up / n_samples_per_angle
    p_left_down[i] = sum_left_down / n_samples_per_angle
    p_right_up[i] = sum_right_up / n_samples_per_angle
    p_right_down[i] = sum_right_down / n_samples_per_angle

    # calculate correlation probabilities
    p_up_up[i] = sum_up_up / n_samples_per_angle
    p_up_down[i] = sum_up_down / n_samples_per_angle
    p_down_up[i] = sum_down_up / n_samples_per_angle
    p_down_down[i] = sum_down_down / n_samples_per_angle

# plots
for i in range(n_angles):
    # plots left and right
    plt.figure("left up")
    plt.plot(alpha[i], p_left_up[i], "ro")

    plt.figure("left down")
    plt.plot(alpha[i], p_left_down[i], "ro")

    plt.figure("right up")
    plt.plot(alpha[i], p_right_up[i], "ro")

    plt.figure("right down")
    plt.plot(alpha[i], p_right_down[i], "ro")

    # plot correlation
    plt.figure("up up")
    plt.plot(alpha[i], p_up_up[i], "ro")

    plt.figure("up down")
    plt.plot(alpha[i], p_up_down[i], "ro")

    plt.figure("down up")
    plt.plot(alpha[i], p_down_up[i], "ro")

    plt.figure("down down")
    plt.plot(alpha[i], p_down_down[i], "ro")

# finish plots
plt.figure("left up")
plt.title("left up")
plt.gca().set_ylim([0, 1])
plt.xlabel("angle between measurement axis")
plt.ylabel("probability")
y = 1 / 2 * np.ones_like(alpha)
plt.plot(alpha, y, linewidth=3)

plt.figure("left down")
plt.title("left down")
plt.gca().set_ylim([0, 1])
plt.xlabel("angle between measurement axis")
plt.ylabel("probability")
y = 1 / 2 * np.ones_like(alpha)
plt.plot(alpha, y, linewidth=3)

plt.figure("right up")
plt.title("right up")
plt.gca().set_ylim([0, 1])
plt.xlabel("angle between measurement axis")
plt.ylabel("probability")
y = 1 / 2 * np.ones_like(alpha)
plt.plot(alpha, y, linewidth=3)

plt.figure("right down")
plt.title("right down")
plt.gca().set_ylim([0, 1])
plt.xlabel("angle between measurement axis")
plt.ylabel("probability")
y = 1 / 2 * np.ones_like(alpha)
plt.plot(alpha, y, linewidth=3)

plt.figure("up up")
plt.title("p_up_up")
plt.xlabel("angle between measurement axis")
plt.ylabel("probability")
y = 1 / 4 * (1 - np.cos(alpha))
plt.plot(alpha, y, linewidth=3)

plt.figure("up down")
plt.title("p_up_down")
plt.xlabel("angle between measurement axis")
plt.ylabel("probability")
y = 1 / 4 * (1 + np.cos(alpha))
plt.plot(alpha, y, linewidth=3)

plt.figure("down up")
plt.title("p_down_up")
plt.xlabel("angle between measurement axis")
plt.ylabel("probability")
y = 1 / 4 * (1 + np.cos(alpha))
plt.plot(alpha, y, linewidth=3)

plt.figure("down down")
plt.title("p_down_down")
plt.xlabel("angle between measurement axis")
plt.ylabel("probability")
y = 1 / 4 * (1 - np.cos(alpha))
plt.plot(alpha, y, linewidth=3)

plt.show()
