import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init
from memory import Memory
from params import Params
import os

from torch.distributions import MultivariateNormal, Categorical

class ActorNetwork(nn.Module):
    def __init__(self, params:Params):
        super(ActorNetwork, self).__init__()

        # shortcut to parameters
        p = params
        self.device = torch.device('cuda' if p.use_cuda else 'cpu')

        self.checkpoint_file = os.path.join(
            p.checkpoint_dir, p.model_name+'_actor'
        )
        self.actions_continuous = p.actions_continuous

        self.action_size = p.action_size
        self.set_action_std(p.action_std)

        if self.actions_continuous: 
            self.actor = nn.Sequential(
                nn.Linear(p.state_size, p.hidden_size1),
                nn.ReLU(),
                nn.Linear(p.hidden_size1, p.hidden_size2),
                nn.ReLU(),
                nn.Linear(p.hidden_size2, p.action_size),
                nn.Tanh()
            ).float()
        else:
            self.actor = nn.Sequential(
                nn.Linear(p.state_size, p.hidden_size1),
                nn.ReLU(),
                nn.Linear(p.hidden_size1, p.hidden_size2),
                nn.ReLU(),
                nn.Linear(p.hidden_size2, p.action_size),
                nn.Softmax(dim=-1)
            ).float()

        self.optimizer = optim.Adam(self.parameters(), lr=p.learning_rate)
        self.to(self.device)

    def cov_mat(self, action_mean):
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        return cov_mat

    def forward(self, state):
        if self.actions_continuous:
            action_mean = self.actor(state)
            cov_mat = self.cov_mat(action_mean)
            distribution = MultivariateNormal(action_mean, cov_mat)
        
        else:
            action_probs = self.actor(state)
            distribution = Categorical(action_probs)

        return distribution

    def get_action(self, state):
        distribution = self.forward(state)

        action = distribution.sample()
        action_logprob = distribution.log_prob(action)

        return action, action_logprob

    def calculate_entropy(self, state, action):
        distribution = self.forward(state)

        action_logprobs = distribution.log_prob(action)
        dist_entropy = distribution.entropy()

        return action_logprobs, dist_entropy

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_size,), new_action_std * new_action_std).to(self.device)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class PredictorNetwork(nn.Module):
    def __init__(self, params:Params):
        super(PredictorNetwork, self).__init__()
        p = params

        self.checkpoint_file = os.path.join(
            p.checkpoint_dir, p.model_name+'_predictor'
        )
        self.predictor = nn.Sequential(
                nn.Linear(p.state_size+p.action_size, p.hidden_size1),
                nn.LeakyReLU(),
                nn.Linear(p.hidden_size1, p.hidden_size2),
                nn.LeakyReLU(),
                nn.Linear(p.hidden_size2, p.state_size),
        ).float()

        self.optimizer = optim.Adam(self.parameters(), lr=p.learning_rate)
        self.device = torch.device('cuda' if p.use_cuda else 'cpu')
        self.to(self.device)

    def forward(self, prev_states, actions, dim=-1):
        inputs =  torch.cat((prev_states, actions), dim).float()
        pred_next_states = self.predictor(inputs) + prev_states
        return pred_next_states

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, params:Params):
        super(CriticNetwork, self).__init__()
        p = params

        self.checkpoint_file = os.path.join(
            p.checkpoint_dir, p.model_name+'_critic'
        )
        self.critic = nn.Sequential(
                nn.Linear(p.state_size, p.hidden_size1),
                nn.ReLU(),
                nn.Linear(p.hidden_size1, p.hidden_size2),
                nn.ReLU(),
                nn.Linear(p.hidden_size2, 1)
        ).float()

        self.optimizer = optim.Adam(self.parameters(), lr=p.learning_rate)
        self.device = torch.device('cuda' if p.use_cuda else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))