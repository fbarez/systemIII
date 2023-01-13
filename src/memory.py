""" Defines the Memory class, which saves data at every timestep for the
given run, and processes the returned values into advantage arrays.
"""
# pylint: disable=attribute-defined-outside-init
from typing import Optional
import numpy as np
from regex import W
import torch
from torch import Tensor
from scipy.signal import lfilter
from params import Params

def tensorify(array):
    return torch.stack([torch.tensor(a) for a in array]).float()

def numpify(array):
    return np.array([np.array(a) for a in array], dtype=np.float32)

class Memory:
    """ Memory class which saves data at every timestep for the given run.
    Also processes the returned values into advantages.
    """
    def __init__(self, params:Params = None, use_cuda = None, state_mapping = None):
        if params is None:
            params = Params(0,0)
        self.params = params
        assert isinstance(self.params, Params)

        self.state_mapping = state_mapping
        use_cuda = torch.cuda.is_available() if use_cuda is None else use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.clear_memory()

    def safety_check(self):
        # basic sanity check, to make sure all arrays are initialized correctly
        # ensure all arrays that recored each timestep are equal length
        timestep_arrays = [
            self.curr_states,
            self.next_states,
            self.pred_states,
            self.action_means,
            self.actions,
            self.logprobs,
            self.rewards,
            self.values,
            self.costs,
            self.cost_values,
            self.dones,
            self.infos
        ]

        if self.advantages_calculated:
            timestep_arrays += [
                self.advantages,
                self.returns,
                self.cost_advantages,
                self.cost_returns,
            ]

        n_timesteps = len(timestep_arrays[0])
        for a in timestep_arrays:
            assert len(a) == n_timesteps

        # ensure all arrays that recored each episode are equal length
        episode_arrays = [
            self.done_indices[:-1],
            self.episode_costs,
            self.episode_rewards,
        ]

        n_episodes = len(self.done_indices[:-1])
        for a in episode_arrays:
            assert len(a) == n_episodes

        return True

    def calculate_advantages(self):
        calculate_advantages(self)

    def normalize_advantages(self):
        # Use normalization / rescaling of the advantage values
        eps = self.params.normalization_epsilon
        self.advantages = np.array(self.advantages)
        with torch.no_grad():
            adv_mean = self.advantages.mean()
            adv_std  = self.advantages.std()
        self.advantages_scaling = (adv_std + eps)
        self.advantages -= adv_mean
        self.advantages /= self.advantages_scaling


        # Center, but do NOT rescale advantages for cost gradient
        cost_adv_mean = self.cost_advantages.mean()
        self.cost_advantages -= cost_adv_mean

        self.advantages_calculated = True

    def prepare(self):
        self.calculate_advantages()
        if self.params.normalize_advantages:
            self.normalize_advantages()

        # arrays that record data for each timestep
        self.curr_states = torch.stack(self.curr_states).to(self.device)
        self.next_states = torch.stack(self.next_states).to(self.device)
        self.pred_states = torch.stack(self.pred_states).to(self.device)
        self.actions = torch.stack(self.actions).to(self.device)
        self.logprobs = torch.stack(self.logprobs).to(self.device)
        self.action_means = torch.stack(self.action_means).to(self.device)
        self.rewards = tensorify(self.rewards)
        self.values = torch.tensor(self.values).to(self.device)
        self.costs = torch.stack(self.costs).to(self.device)
        self.cost_values = torch.tensor(self.cost_values).to(self.device)
        self.dones = np.array(self.dones)
        self.infos = self.infos

        if self.advantages_calculated:
            self.advantages = torch.tensor(self.advantages.copy())
            self.returns = torch.tensor(self.returns.copy())
            self.cost_advantages = torch.tensor(self.cost_advantages.copy())
            self.cost_returns = torch.tensor(self.cost_returns.copy())

        # arrays that record data for each episode
        self.done_indices = np.array(self.done_indices)
        self.episode_costs = np.array(self.episode_costs[:-1])
        self.episode_rewards = np.array(self.episode_rewards[:-1])

        # basic check to make sure that each array has the correct number of items
        self.safety_check()

        return self

    def add(self, curr_state, next_state, pred_state, action_mean, action,
            action_logprob, reward, value, cost, cost_value, done, info):
        # Note that here we should only add states that are already flattened
        self.curr_states.append(curr_state)
        self.next_states.append(next_state)
        self.pred_states.append(pred_state)
        self.action_means.append(action_mean)
        self.actions.append(action)
        self.logprobs.append(action_logprob)
        self.rewards.append(reward)
        self.values.append(value.item())
        self.costs.append(cost)
        self.cost_values.append(cost_value.item())
        self.dones.append(done)

        # episode values
        self.episode_costs[-1] += cost
        self.episode_rewards[-1] += reward

        if done:
            self.done_indices.append(len(self.dones)-1)
            self.episode_costs.append(0)
            self.episode_rewards.append(0)
            # self.calculate_advantages()

        # List[dict]
        self.infos.append(info)

        return self

    def clear_memory(self):
        self.curr_states = []
        self.next_states = []
        self.pred_states = []
        self.action_means = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.costs = []
        self.cost_values = []
        self.dones = []
        self.infos = []

        # episode arrays
        self.done_indices = [0]
        self.episode_costs = [0]
        self.episode_rewards = [0]

        # calculated arrays
        self.advantages_calculated = False
        self.advantages = np.array([], dtype=np.float32)
        self.returns = np.array([], dtype=np.float32)
        self.cost_advantages = np.array([], dtype=np.float32)
        self.cost_returns = np.array([], dtype=np.float32)

        self.advantages_scaling = 1.
        self.cost_advantages_scaling = 1.

        return self

    def flat_get(self, current_state_flat:torch.Tensor, key:str):
        if not self.state_mapping:
            raise Exception("mapping not defined")

        return current_state_flat[self.state_mapping[key][0]:self.state_mapping[key][1]]

    def map_and_flatten(self, state:torch.Tensor):
        state_flat, state_mapping = map_and_flatten_state(state)
        self.state_mapping = state_mapping
        return state_flat, state_mapping

    def flatten_state(self, state:torch.Tensor, return_mapping:bool=False ):
        if return_mapping:
            return self.map_and_flatten(state)

        if self.state_mapping is None:
            return self.map_and_flatten(state)[0]

        if isinstance(state, np.ndarray):
            return torch.tensor(state).float().to(self.device)

        state_flat = []
        for item in state.values():
            state_flat.extend(item.flatten())
        state_flat = torch.tensor(state_flat).float().to(self.device)

        return state_flat

def map_and_flatten_state(state:torch.Tensor):
    state_mapping = {}
    state_flat = []
    if isinstance(state, np.ndarray):
        return torch.tensor(state).float(), state_mapping

    for key, item in state.items():
        starting_index = len(state_flat)
        state_flat.extend( item.flatten() )
        ending_index = len(state_flat)
        state_mapping[key] = [starting_index, ending_index]

    state_flat = torch.tensor(state_flat).float()

    return state_flat, state_mapping

# Calculate Advantages
#########################################################################################
# -> advantages[0] == rewards[0] +       g   * values[1] -             values[0]
#      +  g   * l   * rewards[1] +  l  * g^2 * values[2] - g   * l   * values[1]
#      +  g^2 * l^2 * rewards[2] + l^2 * g^3 * values[2] - g^2 * l^2 * values[2]
#      + ...

# -> returns[0]    == rewards[0] + g * rewards[1] + g^2 * rewards[2] + ...

def calculate_advantages(memory):
    """ Calculates advantages for reward and cost.
    Partially copied from:
    https://github.com/openai/safety-starter-agents/blob/master/safe_rl/pg/buffer.py
    """
    # extract necessary parameters
    reward_decay = memory.params.reward_decay
    gae_lambda   = memory.params.gae_lambda
    cost_decay   = memory.params.cost_decay
    cost_lambda  = memory.params.cost_lambda

    # get arrays
    rewards      = memory.rewards
    values       = memory.values
    costs        = memory.costs
    cost_values  = memory.cost_values
    dones        = memory.dones
    done_indices = memory.done_indices

    # Make sure arrays are in the correct format
    rewards      = numpify( rewards ).squeeze()
    values       = numpify( values ).squeeze()
    costs        = numpify( costs ).squeeze()
    cost_values  = numpify( cost_values ).squeeze()

    # calculate advantages
    advantage_method = "separate"

    # advantage method used in safety-starter-agent
    if advantage_method == "separate":
        memory.advantages, memory.returns = calculate_advantages_separate(
            done_indices, rewards, values, reward_decay, gae_lambda
        )
        memory.cost_advantages, memory.cost_returns = calculate_advantages_separate(
            done_indices, costs, cost_values, cost_decay, cost_lambda
        )

    # advantage method used in reference PPO implementation I initially used
    if advantage_method == "together":
        memory.advantages, memory.returns = calculate_advantages_together(
            rewards, values, dones, reward_decay, gae_lambda
        )
        memory.cost_advantages, memory.cost_returns = calculate_advantages_together(
            costs, cost_values, dones, cost_decay, cost_lambda
        )

    return memory

# Essentially the way advantages and returns are computed in safety-starter-agents
def calculate_advantages_separate(_done_indices, _rewards, _values, _decay, _lambda):
    done_indices = np.append(_done_indices, len(_rewards))
    start_end = zip( done_indices[:-1], done_indices[1:] )

    advantages = np.array([], dtype=np.float32)
    returns    = np.array([], dtype=np.float32)

    for (start_index, end_index) in start_end:
        if start_index == end_index:
            continue
        rewards = np.append( _rewards[start_index:end_index], 0 )
        values  = np.append(  _values[start_index:end_index], 0 )
        deltas  = rewards[:-1] + _decay * values[1:] - values[:-1]
        ep_adv = discount_cumsum(deltas, _decay*_lambda)
        ep_ret = discount_cumsum(rewards, _decay)[:-1]

        # pylint: disable=unexpected-keyword-arg
        advantages = np.concatenate([ advantages, ep_adv ], dtype=np.float32)
        returns    = np.concatenate([ returns,    ep_ret ], dtype=np.float32)

    return advantages, returns

# The way advantages are calculated in my previous implementation
def calculate_advantages_together(rewards, values, dones, decay, _lambda):
    cumulative_rewards = \
        discount_cumsum_rewards( rewards, decay*_lambda )
    cumulative_values  = \
        discount_cumsum_values(  values, dones, decay, _lambda )

    advantages = cumulative_rewards - cumulative_values

    return advantages, cumulative_rewards

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    x_reversed = np.array( x )[::-1]
    return lfilter([1], [1, float(-discount)], x_reversed, axis=0)[::-1]

def discount_cumsum_rewards( rewards, discount ):
    cumulative_rewards = np.zeros(len(rewards), dtype=np.float32)
    cumulative_rewards[-1] = rewards[-1]

    for t in range( len(rewards)-2, -1, -1 ):
        cumulative_rewards[t] += rewards[t]
        cumulative_rewards[t] += discount * cumulative_rewards[t+1]

    return cumulative_rewards

def discount_cumsum_values( values, dones, decay, _lambda ):
    # lim = params.cumulative_limit

    cumulative_values = np.zeros(len(values), dtype=np.float32)
    cumulative_values[-1] = values[-1]
    for t in range( len(values)-2, -1, -1 ):
        cumulative_values[t] += values[t]
        cumulative_values[t] += (decay*_lambda) * cumulative_values[t+1]
        # cumulative_values[t]  = np.median([ -lim, cumulative_values[t], lim ])
        if not dones[t]:
            cumulative_values[t] -= decay * values[t+1]

    return cumulative_values
