#!/usr/bin/python

# Imports
import numpy as np
import scipy.stats as stats
import pickle
import sys, os
from itertools import *
import pdb
from matplotlib import colors
from pprintpp import pprint
import inspect

from MCMCBanditSampling import *
import deepmind_lab


def action_segments():
	env = deepmind_lab.Lab('tests/empty_room_test', [])
	action_spec = env.action_spec()
	action_index = {action['name']: i for i, action in enumerate(action_spec)}
	pprint(action_index)


if __name__ == '__main__':
	path = os.path.dirname(inspect.getfile(deepmind_lab))
	deepmind_lab.set_runfiles_path(path)
	action_segments()

	# deepmind_lab.set_runfiles_path(runfiles_path)


