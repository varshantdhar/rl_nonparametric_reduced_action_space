#!/usr/bin/python

from agent import *

# Imports: python modules
import abc
import numpy as np
import scipy.stats as stats

######## Helper functions ########
def online_update_mean_var(r, new_instance, this_mean, this_m2):
    this_delta=new_instance - this_mean
    new_mean=this_mean+this_delta/r
    new_m2=this_m2+this_delta*(new_instance-new_mean)

    if r < 2:
        new_var=np.nan
    else:
        new_var=new_m2/(r-1)

    return (new_mean, new_m2, new_var)



class Bandit(abc.ABC, object):
    """Abstract Class for Bandits
    Attributes:
        A: size of the multi-armed bandit
        reward_function: dictionary with information about the reward distribution and its parameters is provided
        context: d_context by (at_least) t_max array with context for every time instant (None if does not apply)
        actions: the actions that the bandit takes (per realization) as A by t_max array
        rewards: rewards obtained by each arm of the bandit (per realization) as A by t_max array
        regrets: regret of the bandit (per realization) as t_max array
        cumregrets: cumulative regret of the bandit (per realization) as t_max array
        rewards_expected: the expected rewards computed for each arm of the bandit (per realization) as A by t_max array
        actions_R: dictionary with the actions that the bandit takes (for R realizations)
        rewards_R: dictionary with the rewards obtained by the bandit (for R realizations)
        regrets_R: dictioinary with the regret of the bandit (for R realizations)
        cumregrets_R: dictioinary with the cumulative regret of the bandit (for R realizations)
        rewards_expected_R: the expected rewards of the bandit (for R realizations)
    """

    def __init__(self, A, reward_function):
        """Initialize the Bandit object and its attributes

        Args:
            A: the size of the bandit
            reward_function: the reward function of the bandit
        """

        # General Attributes
        self.A = A
        self.reward_function = reward_function
        self.context = None

        # Per realization
        self.actions = None
        self.rewards = None
        self.regrets = None
        self.cumregrets = None
        self.rewards_expected = None

        # For all realizations
        self.true_expected_rewards = None
        self.actions_R = None
        self.rewards_R = None
        self.regrets_R = None
        self.cumregrets_R = None
        self.rewards_expected_R = None

    def play_arm(self, a, t):
        """Play bandit's arm a with true reward function

        Args:
            a: arm to play
            t: time index (or set of indexes)
        """
        dqn_agent = agent.DQN()
        dqn_agent.step(a)
        self.rewards[a, t] = np.nan
        return

    def compute_true_expected_rewards(self):
        """Compute the expected rewards of the bandit for the true reward function

        Args:
            None
        """
        self.true_expected_rewards = np.einsum(
            "ak,akd,dt->at",
            self.reward_function["pi"],
            self.reward_function["theta"],
            self.context,
        )

