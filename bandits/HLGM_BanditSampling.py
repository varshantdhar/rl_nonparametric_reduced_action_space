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


def get_bandit(A, K, pi, theta, sigma, prior_K, context, d_context):
	reward_function={'pi': pi, 'theta': theta, 'sigma': sigma}

	########## Inference
	# MCMC (Gibbs) parameters
	gibbs_max_iter=4
	gibbs_loglik_eps=0.01
	# Plotting
	gibbs_plot_save='show'
	gibbs_plot_save=None
	if gibbs_plot_save != None and gibbs_plot_save != 'show':
		# Plotting directories
		gibbs_plots=dir_string+'/gibbs_plots'
		os.makedirs(gibbs_plots, exist_ok=True)

	########## Priors
	gamma=0.1
	alpha=1.
	beta=1.
	sigma=1.
	pitman_yor_d=0
	assert (0<=pitman_yor_d) and (pitman_yor_d<1) and (gamma >-pitman_yor_d)

	thompsonSampling={'arm_N_samples':1}

	# Hyperparameters
	# Concentration parameter
	prior_d=pitman_yor_d*np.ones(A)
	prior_gamma=gamma*np.ones(A)
	# NIG for linear Gaussians
	prior_alpha=alpha*np.ones(A)
	prior_beta=beta*np.ones(A)

	# Initial thetas
	prior_theta=np.ones((A,d_context))
	prior_Sigma=np.zeros((A,d_context, d_context))
	# Initial covariances: uncorrelated
	for a in np.arange(A):
		prior_Sigma[a,:,:]=sigma*np.eye(d_context)

	# Reward prior as dictionary
	reward_prior={'d':prior_d, 'gamma':prior_gamma, 'alpha':prior_alpha, 'beta':prior_beta, 'theta':prior_theta, 'Sigma':prior_Sigma, 
		'gibbs_max_iter':gibbs_max_iter, 'gibbs_loglik_eps':gibbs_loglik_eps}

	bandit = MCMCBanditSampling(A, reward_function, reward_prior, thompsonSampling)
	return bandit



if __name__ == "__main__":
    path = os.path.dirname(inspect.getfile(deepmind_lab))
    deepmind_lab.set_runfiles_path(path)

    coords, A = action_segments()  # Rotation Axes, Number of Arms
    K = 2  # Number of Mixtures per arm in the bandit
    prior_K = 2  #

    env = deepmind_lab.Lab(
        "tests/empty_room_test",
        ["RGB_INTERLEAVED"],
        config={"fps": "60", "controls": "external", "width":"80", "height":"80"}
    )

    env.reset()

    rewards = 0
    length = 100
    t_max = 1

    for i in six.moves.range(length):
        if not env.is_running():
            print("Environment stopped early")
            env.reset()
        context = env.observations()["RGB_INTERLEAVED"].flatten()
        d_context = context.shape[0]
        pi = np.random.rand(A, K)
        pi = pi / pi.sum(axis=1, keepdims=True)
        theta = np.random.randn(A, K, d_context)
        sigma=np.ones((A,K))
        bandit = get_bandit(A, K, pi, theta, sigma, prior_K, context, d_context)
        if i == 0:
        	bandit.execute_init(t_max, context)
        bandit.execute(t_max, context)
        bandit.execute_update(t_max, context)
        break
        # action = agent.step(reward, obs["RGB_INTERLEAVED"])
        # reward = env.step(action, 1)
