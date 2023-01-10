import numpy as np
import torch
import scipy

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
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Memory:
    def __init__(self, params, use_cuda=None, state_mapping=None):
        self.state_mapping = state_mapping
        use_cuda = torch.cuda.is_available() if use_cuda is None else use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.params = params

        self.clear_memory()

    def safety_check(self):
        arrays = [
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
            arrays += [
                self.advantages,
                self.returns,
                self.cost_advantages,
                self.cost_returns,
            ]

        L = len(arrays[0])
        for a in arrays:
            assert len(a) == L

        return True

    def tensorify(self, array):
        return torch.stack([torch.tensor(a) for a in array]).float().to(self.device)

    def calculate_advantages(self, last_value=0, last_cost_value=0):
        """ Calculates advantages for reward and cost.
        Copied from:
        https://github.com/openai/safety-starter-agents/blob/master/safe_rl/pg/buffer.py
        """
        # extract necessary parameters
        reward_decay = self.params.reward_decay
        gae_lambda = self.params.gae_lambda
        cost_decay = self.params.cost_decay
        cost_lambda = self.params.cost_lambda

        # Calculate for Values
        rewards = np.append(np.array(self.rewards), last_value)
        values = np.append(np.array(self.values), last_value)
        deltas = rewards[:-1] + self.reward_decay * values[1:] - values[:-1]
        self.advantages = discount_cumsum(deltas, reward_decay*gae_lambda)
        self.returns = discount_cumsum(rewards, self.gamma)[:-1]

        # Calculate for costs/constraints
        costs = np.append(np.array(self.costs), last_cost_value)
        cost_values = np.append(np.array(self.cost_values), last_cost_value)
        cost_deltas = costs[:-1] + self.gamma * cost_values[1:] - cost_values[:-1]
        self.cost_advantages = discount_cumsum(cost_deltas, cost_decay*cost_lambda)
        self.cost_returns = discount_cumsum(costs, self.cost_decay)[:-1]
        
        return self

    def prepare(self, calculate_advantages=False):
        if calculate_advantages:
            self.calculate_advantages()

        self.curr_states = torch.stack(self.curr_states).to(self.device)
        self.next_states = torch.stack(self.next_states).to(self.device)
        self.pred_states = torch.stack(self.pred_states).to(self.device)
        self.actions = torch.stack(self.actions).to(self.device)
        self.logprobs = torch.stack(self.logprobs).to(self.device)
        self.action_means = torch.stack(self.action_means).to(self.device)
        self.rewards = self.tensorify(self.rewards)
        self.values = torch.stack(self.values).to(self.device)
        self.costs = torch.stack(self.costs).to(self.device)
        self.cost_values = torch.stack(self.cost_values).to(self.device)
        self.dones = np.array(self.dones)
        self.infos

        if self.advantages_calculated:
            self.advantages = torch.stack()

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
        self.values.append(value)
        self.costs.append(cost)
        self.cost_values.append(cost)
        self.dones.append(done)
        
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

        # calculated arrays
        self.advantages_calculated = False
        self.advantages = []
        self.returns = []
        self.cost_advantages = []
        self.cost_returns = []

        return self
    
    def flat_get(self, current_state_flat:torch.Tensor, key:str):
        if not self.state_mapping:
            raise Exception("mapping not defined")

        return current_state_flat[self.state_mapping[key][0]:self.state_mapping[key][1]]

    def map_and_flatten(self, state:torch.Tensor):
        state_mapping = {}
        state_flat = []
        if type(state) is np.ndarray:
            return torch.tensor(state).float().to(self.device), state_mapping

        for key, item in state.items():
            starting_index = len(state_flat)
            state_flat.extend( item.flatten() )
            ending_index = len(state_flat)
            state_mapping[key] = [starting_index, ending_index]
        self.state_mapping = state_mapping
        state_flat = torch.tensor(state_flat).float().to(self.device)

        return state_flat, state_mapping

    def flatten_state(self, state:torch.Tensor, return_mapping:bool=False ):
        if return_mapping:
            return self.map_and_flatten(state)
        
        if self.state_mapping is None:
            return self.map_and_flatten(state)[0]

        if type(state) is np.ndarray:
            return torch.tensor(state).float().to(self.device)

        state_flat = []
        for item in state.values():
            state_flat.extend(item.flatten())
        state_flat = torch.tensor(state_flat).float().to(self.device)

        return state_flat
