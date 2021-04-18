#!/usr/bin/python

# Imports
from MCMCBanditSampling import *
from agents import agent
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


def get_bandit(A, K, pi, theta, sigma, prior_K, d_context):
	reward_function={'pi': pi, 'theta': theta, 'sigma': sigma}

	########## Inference
	# MCMC (Gibbs) parameters
	gibbs_max_iter=4
	gibbs_loglik_eps=0.01

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

    coords, A = agent.action_segments()  # Rotation Axes, Number of Arms
    rewards = 0
    R = 10 # Number of realizations to run
    t_max = 100 # Time-instants to run the bandit
    width = 8
    height = 8
    d_context = width * height * 3 # Context dimension

    K = 2 # Number of mixtures per arm of the bandit
    prior_K = 2  # Assumed prior number of mixtures (per arm)
    pi = np.random.rand(A, K)  
    pi = pi / pi.sum(axis=1, keepdims=True) # Mixture proportions per arm
    theta = np.random.randn(A, K, d_context) # Thetas per arm and mixtures
    sigma=np.ones((A,K)) # Variances per arm and mixtures

    bandit = get_bandit(A, K, pi, theta, sigma, prior_K, d_context)

    env = deepmind_lab.Lab(
        "tests/empty_room_test",
        ["RGB_INTERLEAVED"],
        config={"fps": "60", "controls": "external", "width":"8", "height":"8"}
    )
    env.reset()

    bandit.execute_realizations(R, t_max, env, d_context)

    # action = agent.step(reward, obs["RGB_INTERLEAVED"])
    # reward = env.step(action, 1)
