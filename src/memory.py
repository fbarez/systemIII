import numpy as np
import torch

class Memory:
    def __init__(self, use_cuda=False, state_mapping=None):
        self.state_mapping = state_mapping
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.curr_states = []
        self.next_states = []
        self.pred_states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def safety_check(self):
        arrays = [
            self.curr_states,
            self.next_states,
            self.pred_states,
            self.actions,
            self.logprobs,
            self.rewards,
            self.values,
            self.dones
        ]
        L = len(arrays[0])
        for a in arrays:
            assert len(a) == L

        return True

    def tensorify(self, array):
        return torch.stack([torch.tensor(a) for a in array]).float().to(self.device)

    def prepare(self):
        self.curr_states = torch.stack(self.curr_states).to(self.device)
        self.next_states = torch.stack(self.next_states).to(self.device)
        self.pred_states = torch.stack(self.pred_states).to(self.device)
        self.actions = torch.stack(self.actions).to(self.device)
        self.logprobs = torch.stack(self.logprobs).to(self.device)
        self.rewards = self.tensorify(self.rewards)
        self.values = torch.stack(self.values).to(self.device)
        self.dones = np.array(self.dones)

        # basic check to make sure that each array has the correct number of items
        self.safety_check()
        return self

    def add(self, curr_state, next_state, pred_state, action, action_logprob, reward, value, done):
        # Note that here we should only add states that are already flattened
        self.curr_states.append(curr_state)
        self.next_states.append(next_state)
        self.pred_states.append(pred_state)
        self.actions.append(action)
        self.logprobs.append(action_logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear_memory(self):
        self.curr_states = []
        self.next_states = []
        self.pred_states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = [] 
    
    def flat_get(self, current_state_flat, key):
        if not self.state_mapping:
            raise Exception("mapping not defined")

        return current_state_flat[self.state_mapping[key][0]:self.state_mapping[key][1]]

    def map_and_flatten(self, state):
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

    def flatten_state(self, state, return_mapping=False ):
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
    