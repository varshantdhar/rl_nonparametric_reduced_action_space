from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from itertools import product
from pprintpp import pprint

import argparse
import random
import numpy as np
import six
import torch


import deepmind_lab
import DQN

def main(env):
	pprint(env.action_spec())
	pprint(env.observation_spec())


if __name__ == '__main__':
	env = deepmind_lab.Lab('lt_chasm', observations,config={'width': '8',    # screen size, in pixels
		'height': '8',   # screen size, in pixels
		'botCount': '1'},  # lt_chasm option.
		renderer='hardware')       # select renderer.
    env.reset()
	main(env)