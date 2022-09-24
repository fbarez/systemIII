import numpy as np
import torch

class Memory:
    def __init__(self, batch_size=64, use_cuda=None, state_mapping=None):
        self.state_mapping = state_mapping
        self.batch_size = batch_size
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.curr_states = []
        self.next_states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def generate_batches(self):
        n_states = len(self.curr_states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return  torch.stack(self.curr_states).to(self.device),\
                torch.stack(self.next_states).to(self.device),\
                torch.stack(self.actions).to(self.device),\
                torch.stack(self.logprobs).to(self.device),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def add(self, curr_state_flat, next_state_flat, action, action_logprob, reward, done):
        self.curr_states.append(curr_state_flat)
        self.next_states.append(next_state_flat)
        self.actions.append(action)
        self.logprobs.append(action_logprob)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.curr_states = []
        self.next_states = []
        self.actions = []
        self.action_logprob = []
        self.rewards = []
        self.dones = []
   
    
    def flat_get(self, current_state_flat, key):
        if not self.state_mapping:
            raise Exception("mapping not defined")

        return current_state_flat[self.state_mapping[key][0]:self.state_mapping[key][1]]

    def map_and_flatten(self, state):
        state_mapping = {}
        state_flat = []
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

        state_flat = []
        for item in state.values():
            state_flat.extend(item.flatten())
        state_flat = torch.tensor(state_flat).float().to(self.device)
        return state_flat

    def to_flatten_list(self, l):
        flat_list = []
        try:
            for element in l:
                try:
                    len(element)
                    flat_list.append(element)
                except:
                    flat_list.extend(self.to_flatten_list(element))
        except:
            return [l]
                
        return flat_list

    