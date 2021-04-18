from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from itertools import product

import argparse
import random
import numpy as np
import six

import deepmind_lab

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
		arr1 = [-1, 0, 1]
		arr2 = [0, 1]
		permutations = list(product(arr1, arr1, arr2, arr2, arr2))
		action_list = []
		for perm in permutations:
			print(perm)
			# action_list.append[coordinates + perm]
		return action_list

	def step(self, a):
		print(self.action_list(a))
		return
