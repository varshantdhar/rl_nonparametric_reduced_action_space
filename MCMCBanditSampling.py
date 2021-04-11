
#!/usr/bin/python

# Imports: python modules
# Imports: other modules
from BanditSampling import * 
from MCMCPosterior import *

class MCMCBanditSampling(BanditSampling, MCMCPosterior):
    """ Class for bandits with sampling policies
            - MCMC parameter posterior updates, based on mixture model approximation to reward posteriors
            - Bandits decide which arm to play by drawing arm candidates from a predictive arm posterior
            - Different arm-sampling approaches are considered
            - Different arm predictive density computation approaches are considered
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
        super().__init__(A, reward_function, reward_prior, arm_predictive_policy)
            
    def compute_arm_predictive_density(self, t):
        """ Method to compute the predictive density of each arm, based on available information at time t:
                Variational parameter posterior updates
                           
            - Monte Carlo over Arms
                - Draw parameters from the posterior
                - Compute the expected reward for each parameter sample
                - Decide, for each sample, which arm is the best
                - Compute the arm predictive density, as a Monte Carlo over best arm samples
        Args:
            t: time of the execution of the bandit
        """
        rewards_expected_samples=np.zeros((self.A, self.arm_predictive_policy['M']))

        for a in np.arange(self.A):
            K_a=self.reward_posterior['K'][a]                    
            # Assignment Suff statistics
            N_ak=np.zeros(K_a)

            # Rewards
            if self.arm_predictive_policy['MC_type'] == 'MC_expectedRewards' or self.arm_predictive_policy['MC_type'] == 'MC_arms':
                if self.reward_prior['K'] != 'nonparametric':
                    rewards_expected_per_mixture_samples=np.zeros((K_a, self.arm_predictive_policy['M']))
                elif self.reward_prior['K'] == 'nonparametric':
                    rewards_expected_per_mixture_samples=np.zeros((K_a+1, self.arm_predictive_policy['M']))

            elif self.arm_predictive_policy['MC_type'] == 'MC_rewards':
                if self.reward_prior['K'] != 'nonparametric':
                    rewards_per_mixture_samples=np.zeros((K_a, self.arm_predictive_policy['M']))
                elif self.reward_prior['K'] == 'nonparametric':
                    rewards_per_mixture_samples=np.zeros((K_a+1, self.arm_predictive_policy['M']))
                
            # Compute for each mixture
            for k in np.arange(K_a):
                # Number of assignments
                N_ak[k]=(self.reward_posterior['Z'][a]==k).sum()

                # Sampling
                # First sample variance from inverse gamma for each mixture
                sigma_samples=stats.invgamma.rvs(self.reward_posterior['alpha'][a,k], scale=self.reward_posterior['beta'][a,k], size=(1,self.arm_predictive_policy['M']))
                # Then multivariate Gaussian parameters
                theta_samples=self.reward_posterior['theta'][a,k,:][:,None]+np.sqrt(sigma_samples)*(stats.multivariate_normal.rvs(cov=self.reward_posterior['Sigma'][a,k,:,:], size=self.arm_predictive_policy['M']).reshape(self.arm_predictive_policy['M'],self.d_context).T)
                
                # Compute expected reward per mixture, linearly combining context and sampled parameters
                rewards_expected_per_mixture_samples[k,:]=np.dot(self.context[:,t], theta_samples)
                
            if self.reward_prior['K'] == 'nonparametric':
                # New Mixture sampling
                # First sample variance from inverse gamma for each mixture
                sigma_samples=stats.invgamma.rvs(self.reward_prior['alpha'][a], scale=self.reward_prior['beta'][a], size=(1,self.arm_predictive_policy['M']))
                # Then multivariate Gaussian parameters
                theta_samples=self.reward_prior['theta'][a,:][:,None]+np.sqrt(sigma_samples)*(stats.multivariate_normal.rvs(cov=self.reward_prior['Sigma'][a,:,:], size=self.arm_predictive_policy['M']).reshape(self.arm_predictive_policy['M'],self.d_context).T)
                
                # Compute expected reward per mixture, linearly combining context and sampled parameters
                rewards_expected_per_mixture_samples[-1,:]=np.dot(self.context[:,t], theta_samples)
                                    
                
            ## How to compute (expected) rewards over mixtures
            # Expected pi
            # Computed expected mixture proportions as determined by Dirichlet multinomial
            # From mixture proportions
            if self.reward_prior['K'] != 'nonparametric':
                # Dirichlet multinomial
                pi=(self.reward_prior['gamma'][a]+N_ak)/(self.reward_prior['gamma'][a].sum()+N_ak.sum())
            elif self.reward_prior['K'] == 'nonparametric':
                if self.reward_posterior['K'][a]==0:
                    pi=np.array([(self.reward_prior['gamma'][a]+K_a*self.reward_prior['d'][a])/(N_ak.sum()+self.reward_prior['gamma'][a])])
                else:
                    pi=np.concatenate(((N_ak-self.reward_prior['d'][a])/(N_ak.sum()+self.reward_prior['gamma'][a]), (self.reward_prior['gamma'][a][None]+K_a*self.reward_prior['d'][a])/(N_ak.sum()+self.reward_prior['gamma'][a])), axis=0)
            else:
                raise ValueError('Invalid reward_prior K={}'.format(self.reward_prior['K']))
            
            # Compute expected rewards, by averaging over expected mixture proportions
            rewards_expected_samples[a,:]=np.einsum('k,km->m', pi, rewards_expected_per_mixture_samples)

            


       	# Monte Carlo integration over arm samples
        # Mean times expected reward is maximum
        self.arm_predictive_density['mean'][:,t]=((rewards_expected_samples.argmax(axis=0)[None,:]==np.arange(self.A)[:,None]).astype(int)).mean(axis=1)
        # Variance of times expected reward is maximum
        self.arm_predictive_density['var'][:,t]=((rewards_expected_samples.argmax(axis=0)[None,:]==np.arange(self.A)[:,None]).astype(int)).var(axis=1)
        # Also, compute expected rewards            
        self.rewards_expected[:,t]=rewards_expected_samples.mean(axis=1)


if __name__ == '__main__':
    main()
