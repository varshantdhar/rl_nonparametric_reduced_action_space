#!/usr/bin/python

# Imports: python modules
# Imports: other modules
from Bandit import *


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

    def execute_init(self, t_max, context):
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

    def execute(self, t_max, context):
        """Execute the Bayesian bandit
        Args:
            t_max: maximum execution time for the bandit
            context: d_context by (at_least) t_max array with context for every time instant
        """
        # Contextual bandit
        self.d_context = context.shape[0]
        self.context = context

        # Initialize attributes
        self.actions = np.zeros((self.A, t_max))
        self.rewards = np.zeros((self.A, t_max))
        self.rewards_expected = np.zeros((self.A, t_max))
        self.arm_predictive_density = {
            "mean": np.zeros((self.A, t_max)),
            "var": np.zeros((self.A, t_max)),
        }
        self.arm_N_samples = np.ones(t_max)

        # Initialize reward posterior
        self.init_reward_posterior()

        # Execute the bandit for each time instant
        print("Start running bandit")
        for t in np.arange(t_max):
            print('Running time instant {}'.format(t))

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
            self.play_arm(action, t)

            if np.isnan(self.rewards[action, t]):
                # This instance has not been played, and no parameter update (e.g. for logged data)
                self.actions[action, t] = 0.0
                print("here")
            else:
                # Update parameter posterior
                self.update_reward_posterior(t)

        print("Finished running bandit at {}".format(t))
        # Compute expected rewards with true function
        self.compute_true_expected_rewards()
        # Compute regret
        self.regrets = self.true_expected_rewards.max(axis=0) - self.rewards.sum(axis=0)
        self.cumregrets = self.regrets.cumsum()

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





