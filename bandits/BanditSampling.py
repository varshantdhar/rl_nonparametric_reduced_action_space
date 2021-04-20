#!/usr/bin/python

# Imports: python modules
# Imports: other modules
from Bandit import *

import DQN
import agent
import time

class BanditSampling(Bandit):
    """Abstract class for bandits with sampling policies

    Attributes (besides inherited):
        reward_prior: the assumed prior for the multi-armed bandit's reward function
        reward_posterior: the posterior for the learned multi-armed bandit's reward function
        arm_predictive_policy: how to compute arm predictive density and sampling policy
        arm_predictive_density: predictive density of each arm
        arm_N_samples: number of candidate arm samples to draw at each time instant
    """

    def __init__(self, A, reward_function, reward_prior, arm_predictive_policy):
        """Initialize the Bandit object and its attributes

        Args:
            A: the size of the bandit
            reward_function: the reward function of the bandit
            reward_prior: the assumed prior for the multi-armed bandit's reward function
            arm_predictive_policy: how to compute arm predictive density and sampling policy
        """

        # Initialize
        super().__init__(A, reward_function)

        # Reward prior
        self.reward_prior = reward_prior
        # Arm predictive computation strategy
        self.arm_predictive_policy = arm_predictive_policy

    def execute_realizations(self, R, t_max, env, d_context):
        """ Execute R realizations of the bandit
        Args:
            R: number of realizations to run
            t_max: maximum execution time for the bandit
            context: d_context by (at_least) t_max array with context for every time instant (None if does not apply)
            exec_type: batch (keep data from all realizations) or sequential (update mean and variance of realizations data)
        """

        # Allocate overall variables
        self.rewards_R={'mean':np.zeros((1,t_max)), 'm2':np.zeros((1,t_max)), 'var':np.zeros((1,t_max))}
        self.regrets_R={'mean':np.zeros((1,t_max)), 'm2':np.zeros((1,t_max)), 'var':np.zeros((1,t_max))}
        self.cumregrets_R={'mean':np.zeros((1,t_max)), 'm2':np.zeros((1,t_max)), 'var':np.zeros((1,t_max))}
        self.rewards_expected_R={'mean':np.zeros((self.A,t_max)), 'm2':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
        self.actions_R={'mean':np.zeros((self.A,t_max)), 'm2':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
        self.arm_predictive_density_R={'mean':np.zeros((self.A,t_max)), 'm2':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
        self.arm_N_samples_R={'mean':np.zeros(t_max), 'm2':np.zeros(t_max), 'var':np.zeros(t_max)}

        # Initialize target and value networks for Deep Q-Learning
        action_dim = 7
        num_actions = 72
        val_model = DQN.Q_NN_multidim(d_context, action_dim, num_actions, num_hidden=20)
        targ_model = DQN.Q_NN_multidim(d_context, action_dim, num_actions, num_hidden=20)
        # Initialize agent
        dqn_agent = agent.QLearning_Agent()

        for r in np.arange(R):
            # Run one realization
            print('Executing realization {}'.format(r))

            # Contextual bandit
            self.d_context = d_context
            self.context = np.zeros((d_context, t_max))

            self.execute(t_max, env, val_model, targ_model, dqn_agent)

            self.rewards_R['mean'], self.rewards_R['m2'], self.rewards_R['var']=online_update_mean_var(r+1, self.rewards.sum(axis=0), self.rewards_R['mean'], self.rewards_R['m2'])
            self.regrets_R['mean'], self.regrets_R['m2'], self.regrets_R['var']=online_update_mean_var(r+1, self.regrets, self.regrets_R['mean'], self.regrets_R['m2'])
            self.cumregrets_R['mean'], self.cumregrets_R['m2'], self.cumregrets_R['var']=online_update_mean_var(r+1, self.cumregrets, self.cumregrets_R['mean'], self.cumregrets_R['m2'])
            self.rewards_expected_R['mean'], self.rewards_expected_R['m2'], self.rewards_expected_R['var']=online_update_mean_var(r+1, self.rewards_expected, self.rewards_expected_R['mean'], self.rewards_expected_R['m2'])
            self.actions_R['mean'], self.actions_R['m2'], self.actions_R['var']=online_update_mean_var(r+1, self.actions, self.actions_R['mean'], self.actions_R['m2'])
            self.arm_predictive_density_R['mean'], self.arm_predictive_density_R['m2'], self.arm_predictive_density_R['var']=online_update_mean_var(r+1, self.arm_predictive_density['mean'], self.arm_predictive_density_R['mean'], self.arm_predictive_density_R['m2'])
            self.arm_N_samples_R['mean'], self.arm_N_samples_R['m2'], self.arm_N_samples_R['var']=online_update_mean_var(r+1, self.arm_N_samples, self.arm_N_samples_R['mean'], self.arm_N_samples_R['m2'])

    def execute(self, t_max, env, val_model, targ_model, dqn_agent):
        """Execute the Bayesian bandit
        Args:
            t_max: maximum execution time for the bandit
            context: d_context by (at_least) t_max array with context for every time instant
        """

        # Initialize attributes
        self.actions = np.zeros((self.A, t_max))
        self.rewards = np.zeros((self.A, t_max))
        self.rewards_expected = np.zeros((self.A, t_max))
        self.arm_predictive_density = {
            "mean": np.zeros((self.A, t_max)),
            "var": np.zeros((self.A, t_max))
        }
        self.arm_N_samples = np.ones(t_max)

        # Initialize reward posterior
        self.init_reward_posterior()

        # Execute the bandit for each time instant
        print("Running bandit to select action set")
        env.reset()
        t = 0

        while env.is_running() and t < t_max:
            start_time = time.time()
            context_ = env.observations()["RGB_INTERLEAVED"].flatten()
            self.context[:,t] = context_

            # Compute predictive density for each arm
            self.compute_arm_predictive_density(t)

            # Compute number of candidate arm samples, based on sampling strategy
            self.arm_N_samples[t] = self.arm_predictive_policy["arm_N_samples"]

            # Pick next action
            self.actions[
                np.random.multinomial(
                    1,
                    self.arm_predictive_density["mean"][:, t],
                    size=int(self.arm_N_samples[t]),
                )
                .sum(axis=0)
                .argmax(),
                t,
            ] = 1
            action = np.where(self.actions[:, t] == 1)[0][0]

            # Play selected arm
            if t == 0:
                print("Running DQN to select action from action set")
            self.play_arm(action, t, env, dqn_agent, self.context, val_model, targ_model)

            if np.isnan(self.rewards[action, t]):
                # This instance has not been played, and no parameter update (e.g. for logged data)
                self.actions[action, t] = 0.0
            else:
                # Update parameter posterior
                if self.rewards[action, t] > 0:
                    print('Reward {} obtained for iteration {}'.format(self.rewards[action, t], t))
                self.update_reward_posterior(t)
            t += 1

        print("Finished running bandit at iteration {}". format(t))
        dqn_agent.q_learning_rewards = 0 # refresh episodic reward count

        # Compute expected rewards with true function
        self.compute_true_expected_rewards()
        # Compute regret
        self.regrets = self.true_expected_rewards.max(axis=0) - self.rewards.sum(axis=0)
        self.cumregrets = self.regrets.cumsum()
        print("Cumulative Regrets for Episode: {}".format(self.cumregrets[-1]))

    @abc.abstractmethod
    def compute_arm_predictive_density(self, t):
        """Abstract method to compute the predictive density of each arm
            It is based on available information at time t, which depends on posterior update type

            Different alternatives on computing the arm predictive density are considered
            - Integration: Monte Carlo
                Due to analitical intractability, resort to MC
                MC over rewards:
                    - Draw parameters from the posterior
                    - Draw rewards, for each parameter sample
                    - Decide, for each drawn reward sample, which arm is the best
                    - Compute the arm predictive density, as a Monte Carlo by averaging best arm samples
                MC over Expectedrewards
                    - Draw parameters from the posterior
                    - Compute the expected reward for each parameter sample
                    - Compute the overall expected reward estimate, as a Monte Carlo by averaging over per-sample expected rewards
                    - Decide, given the MC expected reward, which arm is the best
                MC over Arms
                    - Draw parameters from the posterior
                    - Compute the expected reward for each parameter sample
                    - Decide, for each sample, which arm is the best
                    - Compute the arm predictive density, as a Monte Carlo over best arm samples
        Args:
            t: time of the execution of the bandit
        """


# Making sure the main program is not executed when the module is imported
if __name__ == "__main__":
    main()





