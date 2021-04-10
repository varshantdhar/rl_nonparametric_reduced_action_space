#!/usr/bin/python

# Imports: python modules
# Imports: other modules
from Bandit import * 

class BanditSampling(Bandit):
    """ Abstract class for bandits with sampling policies
    
    Attributes (besides inherited):
        reward_prior: the assumed prior for the multi-armed bandit's reward function
        reward_posterior: the posterior for the learned multi-armed bandit's reward function
        arm_predictive_policy: how to compute arm predictive density and sampling policy
        arm_predictive_density: predictive density of each arm
        arm_N_samples: number of candidate arm samples to draw at each time instant
    """
    
    def __init__(self, A, reward_function, reward_prior, arm_predictive_policy):
        """ Initialize the Bandit object and its attributes
        
        Args:
            A: the size of the bandit
            reward_function: the reward function of the bandit
            reward_prior: the assumed prior for the multi-armed bandit's reward function
            arm_predictive_policy: how to compute arm predictive density and sampling policy
        """
        
        # Initialize
        super().__init__(A, reward_function)
        
        # Reward prior
        self.reward_prior=reward_prior
        # Arm predictive computation strategy
        self.arm_predictive_policy=arm_predictive_policy

    def execute_realizations(self, R, t_max, context=None, exec_type='sequential'):
        """ Execute R realizations of the bandit
        Args:
            R: number of realizations to run
            t_max: maximum execution time for the bandit
            context: d_context by (at_least) t_max array with context for every time instant (None if does not apply)
            exec_type: batch (keep data from all realizations) or sequential (update mean and variance of realizations data)
        """
        self.rewards_R={'mean':np.zeros((1,t_max)), 'm2':np.zeros((1,t_max)), 'var':np.zeros((1,t_max))}
        self.regrets_R={'mean':np.zeros((1,t_max)), 'm2':np.zeros((1,t_max)), 'var':np.zeros((1,t_max))}
        self.cumregrets_R={'mean':np.zeros((1,t_max)), 'm2':np.zeros((1,t_max)), 'var':np.zeros((1,t_max))}
        self.rewards_expected_R={'mean':np.zeros((self.A,t_max)), 'm2':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
        self.actions_R={'mean':np.zeros((self.A,t_max)), 'm2':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
        self.arm_predictive_density_R={'mean':np.zeros((self.A,t_max)), 'm2':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
        self.arm_N_samples_R={'mean':np.zeros(t_max), 'm2':np.zeros(t_max), 'var':np.zeros(t_max)}

        for r in np.arange(R):
            # Run one realization
            print('Executing realization {}'.format(r))
            self.execute(t_max, context)

            self.rewards_R['mean'], self.rewards_R['m2'], self.rewards_R['var']=online_update_mean_var(r+1, self.rewards.sum(axis=0), self.rewards_R['mean'], self.rewards_R['m2'])
            self.regrets_R['mean'], self.regrets_R['m2'], self.regrets_R['var']=online_update_mean_var(r+1, self.regrets, self.regrets_R['mean'], self.regrets_R['m2'])
            self.cumregrets_R['mean'], self.cumregrets_R['m2'], self.cumregrets_R['var']=online_update_mean_var(r+1, self.cumregrets, self.cumregrets_R['mean'], self.cumregrets_R['m2'])
            self.rewards_expected_R['mean'], self.rewards_expected_R['m2'], self.rewards_expected_R['var']=online_update_mean_var(r+1, self.rewards_expected, self.rewards_expected_R['mean'], self.rewards_expected_R['m2'])
            self.actions_R['mean'], self.actions_R['m2'], self.actions_R['var']=online_update_mean_var(r+1, self.actions, self.actions_R['mean'], self.actions_R['m2'])
            self.arm_predictive_density_R['mean'], self.arm_predictive_density_R['m2'], self.arm_predictive_density_R['var']=online_update_mean_var(r+1, self.arm_predictive_density['mean'], self.arm_predictive_density_R['mean'], self.arm_predictive_density_R['m2'])
            self.arm_N_samples_R['mean'], self.arm_N_samples_R['m2'], self.arm_N_samples_R['var']=online_update_mean_var(r+1, self.arm_N_samples, self.arm_N_samples_R['mean'], self.arm_N_samples_R['m2'])





