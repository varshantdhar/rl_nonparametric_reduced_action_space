import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import time
device = "cpu"


def ql(env, agent, context, frame_count, running_rewards):
    # for i in range(n_epoch):
    done = False
    state = agent.init_state(torch.Tensor(context))
    agent.epsilon *= 0.99
    # while not done:
    values, action_ind = agent.get_action(state)
    action = np.array(agent.choose_action(action_ind).numpy(), dtype=np.intc)
    reward = env.step(action, num_steps=4)
    if not env.is_running():
        running_rewards = 0
        done = True
        agent.replay_buffer.add_sample(state, action_ind, reward, state, done)
        agent.train_step(frame_count)
        return (reward, running_rewards)
    next_context = env.observations()["RGB_INTERLEAVED"].flatten()
    next_state = agent.get_state(torch.Tensor(next_context))
        
    agent.replay_buffer.add_sample(state, action_ind, reward, next_state, done)
    agent.train_step(frame_count)
    if reward > 0:
        running_rewards += reward
        print('Score (cumulative rewards): {} '.format(running_rewards))
    return (reward, running_rewards)

class SAValueNN(nn.Module):
    def __init__(self, num_hidden, num_actions):
        super().__init__()
        """
        simple shallow ReLU network:
        
        """
        self.model = nn.Sequential(nn.Linear(1, num_hidden),
                                    nn.ReLU(),
                                    nn.Linear(num_hidden, num_actions))

    def forward(self, state):
        return self.model(state)
    
class Q_NN_multidim(nn.Module):
    def __init__(self, state_dim, action_dim, num_actions, num_hidden=10):
        super().__init__()
        self.state_embed = nn.Sequential(nn.Linear(state_dim, num_hidden),
                                  nn.ReLU(),)
        self.actions_out = [nn.Linear(num_hidden, num_actions) for _ in range(action_dim)]
        
    def forward(self, state):
        embed = self.state_embed(state)
        return torch.stack([v(embed) for v in self.actions_out], dim=1)


class ReplayBuffer:
    def __init__(self, buffer_size, state_size, action_size):
        self.buffer_size = buffer_size
        self.buffer_full = False
        self.place_position = 0
        self.num_in_buffer = 0
        self.state_size = state_size
        self.states = torch.empty(buffer_size, state_size)
        self.next_states = torch.empty(buffer_size, state_size)
        self.action_inds = torch.empty((buffer_size, action_size), dtype=torch.long)
        self.rewards = torch.empty(buffer_size, 1)
        self.done_flags = torch.empty((buffer_size, 1), dtype=torch.bool)
        
    def sample_batch(self, batch_size):
        if self.buffer_full or self.place_position > batch_size:
            ind = np.random.choice(self.num_in_buffer, batch_size, replace=False)
        else:
            ind = np.random.choice(self.place_position, batch_size, replace=True)
            
        return self.states[ind], self.next_states[ind], self.action_inds[ind], self.rewards[ind], self.done_flags[ind]
        
    def add_sample(self, state, action_ind, reward, next_state, done_flag):
        self.states[self.place_position] = state
        self.next_states[self.place_position] = next_state
        self.action_inds[self.place_position] = action_ind
        self.rewards[self.place_position] = reward
        self.done_flags[self.place_position] = done_flag
        
        self.place_position += 1
        if not self.buffer_full:
            self.num_in_buffer += 1
            
        if self.place_position == self.buffer_size:
            self.place_position = 0
            self.buffer_full = True
        return
        
class Q_Learning:
    def __init__(self, epsilon, gamma, value_model, target_model, action_space, state_size, state_scaling=100,
                history_len=3, buffer_size=1000, batch_size=128 ):
        self.epsilon = epsilon
        self.gamma = gamma
        self.value_model = value_model
        self.target_model = target_model
        self.action_space = action_space
        self.num_actions = sum(action_space.shape)
        self.optimizer = torch.optim.Adam(self.value_model.parameters(), lr=0.01)
        self.history_len = history_len
        self.batch_size = batch_size
        
        self.target_model_update_freq = 100
        
        self.replay_buffer = ReplayBuffer(buffer_size, state_size*history_len, action_space.shape[0])
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.value_model.state_dict())
        
    def random_sample(self):
        # for now, assume action space is a vector of actions
        return torch.LongTensor([random.randrange(self.action_space.shape[1]) for _ in range(self.action_space.shape[0])])
        
    def get_action(self, state):
        """
        End goal:
        sample actions given a state and action space
            action space might be a list of actions
        how to define action space?
            maybe easiest is a list of values (since this is a continuous action space)
        """
        q_values = self.value_model(state.unsqueeze(0))
        print(q_values.shape)
        if np.random.random() < self.epsilon:
            rand_ind = self.random_sample()
            return q_values[:, range(self.action_space.shape[0]), rand_ind], rand_ind
        
        # otherwise, compute value for each of these actions
        best_values, best_action_inds = torch.max(q_values, dim=2)
        return best_values, best_action_inds.squeeze(0)
    
    def get_state(self, obs):
        # state is defined as concat of last 4 observations
        _ = self.obs.pop(0)
        self.obs.append(obs)
        return torch.cat(self.obs)
        
    def init_state(self, obs):
        self.obs = [obs for _ in range(self.history_len)]
        return torch.cat(self.obs)
    
    def calc_loss(self, q_values, action_taken_ind, target_values, rewards, done_flags):
        """
        q_value: predicted value for taken action
        target_values: value for each action for update target
        reward: for taken action
        """
        max_target_vals = torch.max(target_values, dim=2).values
        
        q_target = rewards + ~done_flags*self.gamma*max_target_vals

        q_taken = q_values.view(224*4, -1)[range(224*4), action_taken_ind.view(224*4)]
        q_target = q_target.view(224*4)
        return torch.nn.functional.mse_loss(q_taken, q_target)
    
    def update(self, loss):
        loss.backward()
        self.optimizer.step()
    
    def choose_action(self, action_ind):
        return self.action_space[range(self.action_space.shape[0]), action_ind]
    
    def train_step(self, frame_count):
        states, next_states, action_ind, rewards, done_flags = self.replay_buffer.sample_batch(self.batch_size)

        values = self.value_model(states)
        target_values = self.target_model(next_states)

        loss = self.calc_loss(values, action_ind, target_values, rewards, done_flags)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 10)
        
        if frame_count % self.target_model_update_freq == 0:
            self.update_target_model()