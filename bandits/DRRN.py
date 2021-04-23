# from util import *
from collections import namedtuple
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools

######## Helper function ########
def action_segments():
    x = 512
    y = 0
    coords = [[x, y]]
    theta = [(t * np.pi / 180) for t in range(5, 360, 5)]
    for t in theta:
        new_x = np.floor(x * np.cos(t) - y * np.sin(t))
        new_y = np.floor(x * np.sin(t) + y * np.cos(t))
        coords.append((new_x, new_y))
    A = len(coords)
    return coords, A
import numpy as np


def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):
    '''
    Partially borrowed from Keras
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[-maxlen:]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        # post padding
        x[idx, :len(trunc)] = trunc
    return x

def train(agent, state, env, arm, prev_reward=None, prev_action=None, prev_state=None):
    if prev_reward == None:
        rew, prev_action = agent.execute_action(env, state, arm)
        return (rew, prev_action, state)
    else:
        agent.train_network(prev_state, prev_action, prev_reward, state, arm, done = env.is_running())
        rew, prev_action = agent.execute_action(env, state, arm)
        return (rew, prev_action, state)


State = namedtuple('State', ('obs', 'description', 'inventory'))
Transition = namedtuple('Transition', ('state', 'act', 'reward', 'next_state', 'next_acts', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DRRN(torch.nn.Module):
    """
        Deep Reinforcement Relevance Network - He et al. '16
    """
    def __init__(self, action_dim, obs_dim, embedding_dim, hidden_dim):
        super(DRRN, self).__init__()
        self.action_embedding = nn.Embedding(action_dim, embedding_dim)
        self.state_embedding = nn.Embedding(obs_dim, embedding_dim)
        self.state_encoder  = nn.GRU(embedding_dim, hidden_dim)
        # self.look_encoder = nn.GRU(embedding_dim, hidden_dim)
        # self.inv_encoder  = nn.GRU(embedding_dim, hidden_dim)
        self.act_encoder  = nn.GRU(embedding_dim, hidden_dim)
        self.hidden       = nn.Linear(4*hidden_dim, hidden_dim)
        self.act_scorer   = nn.Linear(hidden_dim, 1)


    def packed_rnn(self, x, rnn, type_embed):
        """ Runs the provided rnn on the input x. Takes care of packing/unpacking.
            x: list of unpadded input sequences
            Returns a tensor of size: len(x) x hidden_dim
        """
        lengths = torch.tensor([len(n) for n in x], dtype=torch.long, device="cpu")
        # Sort this batch in descending order by seq length
        lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)
        padded_x = pad_sequences(x)
        x_tt = torch.from_numpy(padded_x).type(torch.long)
        x_tt = x_tt.index_select(0, idx_sort)
        # Run the embedding layer
        if type_embed == 'action':
            embed = self.action_embedding(x_tt).permute(1,0,2) # Time x Batch x EncDim
        elif type_embed == 'state':
            embed = self.state_embedding(x_tt).permute(1,0,2) # Time x Batch x EncDim
        else:
            raise ValueError('Unknown embedding type')
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embed, lengths)
        # Run the RNN
        out, _ = rnn(packed)
        # Unpack
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        # Get the last step of each sequence
        idx = (lengths-1).view(-1,1).expand(len(lengths), out.size(2)).unsqueeze(0)
        out = out.gather(0, idx).squeeze(0)
        # Unsort
        out = out.index_select(0, idx_unsort)
        return out


    def forward(self, state, act_batch):
        """
            Batched forward pass.
            obs_id_batch: iterable of unpadded sequence ids
            act_batch: iterable of lists of unpadded admissible command ids
            Returns a tuple of tensors containing q-values for each item in the batch
        """
        # Zip the state_batch into an easy access format
        # state = State(*zip(*state_batch))
        # This is number of admissible commands in each element of the batch
        state_size = len(state)
        act_sizes = [len(a) for a in act_batch]
        # Combine next actions into one long list
        act_out = self.packed_rnn([act + 512 for act in act_batch], self.act_encoder, 'action')
        # Encode the various aspects of the state
        state_out = self.packed_rnn(state, self.state_encoder, 'state')
        # Expand the state to match the batches of actions
        print(state_out.shape, act_out.shape)
        # state_out = torch.cat([state_out[i].repeat(j,1) for i,j in enumerate(act_sizes)], dim=0)
        z = torch.cat((state_out, act_batch), dim=1) # Concat along hidden_dim
        z = F.relu(self.hidden(z))
        act_values = self.act_scorer(z).squeeze(-1)
        # Split up the q-values by batch
        return act_values.split(act_sizes)


    def act(self, state, act_ids, sample=True):
        """ Returns an action-string, optionally sampling from the distribution
            of Q-Values.
        """
        act_values = self.forward(state, act_ids)
        if sample:
            act_probs = [F.softmax(vals, dim=0) for vals in act_values]
            act_idxs = [torch.multinomial(probs, num_samples=1).item() \
                        for probs in act_probs]
        else:
            act_idxs = [vals.argmax(dim=0).item() for vals in act_values]
        return act_idxs, act_values

class DRRN_Agent:
    def __init__(self):
        self.gamma = 0.9
        self.batch_size = 64
        self.action_dim = 1024
        self.obs_dim = 256
        self.network = DRRN(self.action_dim, self.obs_dim, embedding_dim=128, hidden_dim=128)
        self.memory = ReplayMemory(capacity=5000)
        self.save_path = 'logs'
        self.clip = 5
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=0.0001)

    def action_list(self, a):
        coords, A = action_segments()
        coordinates = (coords[a][0], coords[a][1])
        arr1 = [-1.0, 0.0, 1.0]
        arr2 = [0.0, 1.0]
        permutations = list(itertools.product(arr1, arr1, arr2, arr2, arr2))
        a_list = []
        for perm in permutations:
            a_list.append(np.array(coordinates + perm, dtype=np.intc))
        return a_list

    def execute_action(self, env, state, arm):
        actions = self.action_list(arm)
        # action_space = torch.transpose(torch.Tensor(self.action_list(arm)),0,1)
        action_ids, action_idxs, _ = self.act(state, actions)
        # self.action_space[range(self.action_space.shape[0]), action_ind]
        action_val = [action[idx] for action, idx in zip(actions, action_idxs)]
        action = np.array(action_val.numpy(), dtype=np.intc)
        reward = env.step(action, num_steps=4)
        return (reward, action)

    def train_network(self, context, action, reward, next_state, next_actions, done):
        actions = self.action_list(arm)
        self.observe(state, act, rew, next_state, valids, done)
        loss = self.update()
        if loss is not None:
            print("Obtained a loss!")
            # outfile = open('HLGM_DRRN_LOSS','ab+')
            # pickle.dump({'Loss': loss},outfile)
            # outfile.close()


    def observe(self, state, act, rew, next_state, next_acts, done):
        self.memory.push(state, act, rew, next_state, next_acts, done)


    def act(self, state, poss_acts, sample=True):
        """ Returns a string action from poss_acts. """
        idxs, values = self.network.act(state, poss_acts, sample)
        act_ids = [poss_acts[batch][idx] for batch, idx in enumerate(idxs)]
        return act_ids, idxs, values


    def update(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute Q(s', a') for all a'
        # TODO: Use a target network???
        next_qvals = self.network(batch.next_state, batch.next_acts)
        # Take the max over next q-values
        next_qvals = torch.tensor([vals.max() for vals in next_qvals], device="cpu")
        # Zero all the next_qvals that are done
        next_qvals = next_qvals * (1-torch.tensor(batch.done, dtype=torch.float, device="cpu"))
        targets = torch.tensor(batch.reward, dtype=torch.float, device="cpu") + self.gamma * next_qvals

        # Next compute Q(s, a)
        # Nest each action in a list - so that it becomes the only admissible cmd
        nested_acts = tuple([[a] for a in batch.act])
        qvals = self.network(batch.state, nested_acts)
        # Combine the qvals: Maybe just do a greedy max for generality
        qvals = torch.cat(qvals)

        # Compute Huber loss
        loss = F.smooth_l1_loss(qvals, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.clip)
        self.optimizer.step()
        return loss.item()


    def load(self):
        try:
            self.memory = pickle.load(open(pjoin(self.save_path, 'memory.pkl'), 'rb'))
            self.network = torch.load(pjoin(self.save_path, 'model.pt'))
        except Exception as e:
            print("Error saving model.")
            logging.error(traceback.format_exc())


    def save(self):
        try:
            pickle.dump(self.memory, open(pjoin(self.save_path, 'memory.pkl'), 'wb'))
            torch.save(self.network, pjoin(self.save_path, 'model.pt'))
        except Exception as e:
            print("Error saving model.")
            logging.error(traceback.format_exc())


