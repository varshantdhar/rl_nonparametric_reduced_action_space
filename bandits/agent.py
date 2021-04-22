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
	theta = [(t * np.pi / 180) for t in range(5, 360, 5)]
	for t in theta:
		new_x = np.floor(x * np.cos(t) - y * np.sin(t))
		new_y = np.floor(x * np.sin(t) + y * np.cos(t))
		coords.append((new_x, new_y))
	A = len(coords)
	return coords, A

class QLearning_Agent(object):
	def __init__(self):
		self.q_learning_rewards = 0
		self.QLearning_Buffer = {}

	def action_list(self, a, val_model, targ_model):
		coords, A = action_segments()
		coordinates = (coords[a][0], coords[a][1])
		arr1 = [-1.0, 0.0, 1.0]
		arr2 = [0.0, 1.0]
		permutations = list(product(arr1, arr1, arr2, arr2, arr2))
		a_list = []
		for perm in permutations:
			a_list.append(np.array(coordinates + perm, dtype=np.intc))
		return a_list

	def step(self, a, t, context, env, val_model, targ_model):
		actions = torch.transpose(torch.Tensor(self.action_list(a, val_model, targ_model)),0,1)
		context_size = context.shape[0]
		if str(a) not in self.QLearning_Buffer.keys():
			self.QLearning_Buffer[str(a)] = DQN.Q_Learning(0.5, 0.99, val_model, targ_model, actions, context_size, history_len=1)
		reward, self.q_learning_rewards = DQN.ql(env, self.QLearning_Buffer[str(a)], context[:,t], t, self.q_learning_rewards)
		return reward
