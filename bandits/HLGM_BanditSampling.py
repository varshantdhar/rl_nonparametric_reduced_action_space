#!/usr/bin/python

# Imports
from MCMCBanditSampling import *
# from agent import *
from itertools import *
from matplotlib import colors
from pprintpp import pprint


import numpy as np
import scipy.stats as stats
import pickle
import sys, os
import six
import pdb
import inspect

import deepmind_lab


def action_segments():
    x = 512
    y = 0
    coords = [[x, y]]
    theta = [(t * np.pi / 180) for t in range(15, 360, 15)]
    for t in theta:
        new_x = np.floor(x * np.cos(t) - y * np.sin(t))
        new_y = np.floor(x * np.sin(t) + y * np.cos(t))
        coords.append([new_x, new_y])
    A = len(coords)
    return coords, A


def run(A, K, pi, theta, sigma, prior_K, context):
    return


if __name__ == "__main__":
    path = os.path.dirname(inspect.getfile(deepmind_lab))
    deepmind_lab.set_runfiles_path(path)

    coords, A = action_segments()  # Rotation Axes, Number of Arms
    K = 2  # Number of Mixtures per arm in the bandit
    prior_K = 2  #
    pi = np.random.rand(A, K)
    pi = pi/pi.sum(axis=1, keepdims=True)
    theta = np.random.randn(A, K, args.d_context)

    env = deepmind_lab.Lab(
        "tests/empty_room_test",
        ["RGB_INTERLEAVED"],
        config={"fps": "60", "controls": "external"},
    )

    env.reset()

    rewards = 0

    for _ in six.moves.range(length):
        if not env.is_running():
            print("Environment stopped early")
            env.reset()
        context = env.observations()["RGB_INTERLEAVED"]
        pprint(context.shape)
        break
        run(context)
        action = agent.step(reward, obs["RGB_INTERLEAVED"])
        reward = env.step(action, 1)
