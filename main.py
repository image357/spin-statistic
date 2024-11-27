import matplotlib.pyplot as plt
import numpy as np

from src.general import sphere2cart, spinor
from src.instrument import project_onto_spin_basis
from src.random import coin_flip

n_angles = 100
n_samples_per_angle = 100000

theta = np.linspace(0, np.pi, n_angles)
phi = np.pi / 3
mas = sphere2cart(1, theta, phi)

for i in range(n_angles):
    # sample left "hidden" variables
    left = coin_flip(n_samples_per_angle)

    # (anti-)correlate right and left measurements
    right = -left

    # get spinor
    left_spinor = spinor([0, 0, 1], left)
    right_spinor = spinor([0, 0, 1], right)

    # perform measurement projections
    pj_left = project_onto_spin_basis(left_spinor, mas[i])
    pj_right = project_onto_spin_basis(right_spinor, [0, 0, 1])

    # count left and right measurements
    sum_left_up = np.sum(pj_left == 1)
    sum_left_down = np.sum(pj_left == -1)
    sum_right_up = np.sum(pj_right == 1)
    sum_right_down = np.sum(pj_right == -1)

    # calculate left and right probabilities
    p_left_up = sum_left_up / n_samples_per_angle
    p_left_down = sum_left_down / n_samples_per_angle
    p_right_up = sum_right_up / n_samples_per_angle
    p_right_down = sum_right_down / n_samples_per_angle

    # count correlated measurement occurrence
    sum_up_up = np.logical_and(pj_left == 1, pj_right == 1).sum()
    sum_up_down = np.logical_and(pj_left == 1, pj_right == -1).sum()
    sum_down_up = np.logical_and(pj_left == -1, pj_right == 1).sum()
    sum_down_down = np.logical_and(pj_left == -1, pj_right == -1).sum()

    # calculate correlation probabilities
    p_up_up = sum_up_up / n_samples_per_angle
    p_up_down = sum_up_down / n_samples_per_angle
    p_down_up = sum_down_up / n_samples_per_angle
    p_down_down = sum_down_down / n_samples_per_angle

    # plots left and right
    plt.figure("left up")
    plt.plot(theta[i], p_left_up, "ro")

    plt.figure("left down")
    plt.plot(theta[i], p_left_down, "ro")

    plt.figure("right up")
    plt.plot(theta[i], p_right_up, "ro")

    plt.figure("right down")
    plt.plot(theta[i], p_right_down, "ro")

    # plot correlation
    plt.figure("up up")
    plt.plot(theta[i], p_up_up, "ro")

    plt.figure("up down")
    plt.plot(theta[i], p_up_down, "ro")

    plt.figure("down up")
    plt.plot(theta[i], p_down_up, "ro")

    plt.figure("down down")
    plt.plot(theta[i], p_down_down, "ro")

# finish plots
plt.figure("left up")
plt.title("left up")
plt.gca().set_ylim([0, 1])
plt.xlabel("angle between measurement axis")
plt.ylabel("probability")
y = 1 / 2 * np.ones_like(theta)
plt.plot(theta, y, linewidth=3)

plt.figure("left down")
plt.title("left down")
plt.gca().set_ylim([0, 1])
plt.xlabel("angle between measurement axis")
plt.ylabel("probability")
y = 1 / 2 * np.ones_like(theta)
plt.plot(theta, y, linewidth=3)

plt.figure("right up")
plt.title("right up")
plt.gca().set_ylim([0, 1])
plt.xlabel("angle between measurement axis")
plt.ylabel("probability")
y = 1 / 2 * np.ones_like(theta)
plt.plot(theta, y, linewidth=3)

plt.figure("right down")
plt.title("right down")
plt.gca().set_ylim([0, 1])
plt.xlabel("angle between measurement axis")
plt.ylabel("probability")
y = 1 / 2 * np.ones_like(theta)
plt.plot(theta, y, linewidth=3)

plt.figure("up up")
plt.title("p_up_up")
plt.xlabel("angle between measurement axis")
plt.ylabel("probability")
y = 1 / 4 * (1 - np.cos(theta))
plt.plot(theta, y, linewidth=3)

plt.figure("up down")
plt.title("p_up_down")
plt.xlabel("angle between measurement axis")
plt.ylabel("probability")
y = 1 / 4 * (1 + np.cos(theta))
plt.plot(theta, y, linewidth=3)

plt.figure("down up")
plt.title("p_down_up")
plt.xlabel("angle between measurement axis")
plt.ylabel("probability")
y = 1 / 4 * (1 + np.cos(theta))
plt.plot(theta, y, linewidth=3)

plt.figure("down down")
plt.title("p_down_down")
plt.xlabel("angle between measurement axis")
plt.ylabel("probability")
y = 1 / 4 * (1 - np.cos(theta))
plt.plot(theta, y, linewidth=3)

plt.show()
