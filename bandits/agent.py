from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from itertools import product

import argparse
import random
import numpy as np
import six
import torch

import deepmind_lab
import DQN

######## Helper functions ########
def action_segments():
	x = 512
	y = 0
	coords = [[x, y]]
	theta = [(t * np.pi / 180) for t in range(15, 360, 15)]
	for t in theta:
		new_x = np.floor(x * np.cos(t) - y * np.sin(t))
		new_y = np.floor(x * np.sin(t) + y * np.cos(t))
		coords.append((new_x, new_y))
	A = len(coords)
	return coords, A

class QLearning_Agent(object):
	def action_list(self, a):
		coords, A = action_segments()
		coordinates = (coords[a][0], coords[a][1])
		arr1 = [-1.0, 0.0, 1.0]
		arr2 = [0.0, 1.0]
		permutations = list(product(arr1, arr1, arr2, arr2, arr2))
		a_list = []
		for perm in permutations:
			a_list.append(np.array(coordinates + perm, dtype=np.intc))
		return a_list

	def step(self, a, t, context, env):
		actions = torch.Tensor(self.action_list(a))
		num_actions = len(actions)
		context_size = context.shape[0]
		action_dim = 7
		val_model = DQN.Q_NN_multidim(context_size, 7, num_actions, num_hidden=10)
		targ_model = DQN.Q_NN_multidim(context_size, 7, num_actions, num_hidden=10)
		learner = DQN.Q_Learning(0.5, 0.99, val_model, targ_model, actions, state_size=context_size, history_len=1)
		return DQN.get_reward(env, learner, context[:,t])
