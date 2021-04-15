#!/usr/bin/python

# Imports
import numpy as np
import scipy.stats as stats
import pickle
import sys, os
import pdb
import inspect

import deepmind_lab

from MCMCBanditSampling import *
from itertools import *
from matplotlib import colors
from pprintpp import pprint


def action_segments():
	env = deepmind_lab.Lab('tests/empty_room_test', [])
	action_spec = env.action_spec()
	x = 512
	coords = [x]
	theta = [(t * np.pi / 180) for t in range(45, 360, 45)]
	coords.append([np.floor(x*np.sin(t)) for t in theta])
	pprint(coords)

if __name__ == '__main__':
	path = os.path.dirname(inspect.getfile(deepmind_lab))
	deepmind_lab.set_runfiles_path(path)
	action_segments()

	# deepmind_lab.set_runfiles_path(runfiles_path)


