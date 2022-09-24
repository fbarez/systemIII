import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init
from memory import Memory
import os

from torch.distributions import MultivariateNormal

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, action_std_init, use_cuda, lr,
            hidden_size1=256, hidden_size2=64, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, action_size),
            nn.Tanh()
        ).float().to(self.device)

        self.action_size = action_size
        self.set_action_std(action_std_init)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def cov_mat(self, action_mean):
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        return cov_mat

    def forward(self, state):
        action_mean = self.actor(state)
        cov_mat = self.cov_mat(action_mean)
        distribution = MultivariateNormal(action_mean, cov_mat)

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
    def __init__(self, state_size, action_size, use_cuda, lr,
            hidden_size1=256, hidden_size2=64, chkpt_dir='tmp/ppo'):
        super(PredictorNetwork, self).__init__()

        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.checkpoint_file = os.path.join(chkpt_dir, 'predictor_torch_ppo')
        self.predictor = nn.Sequential(
                nn.Linear(state_size+action_size, hidden_size1),
                nn.LeakyReLU(),
                nn.Linear(hidden_size1, hidden_size2),
                nn.LeakyReLU(),
                nn.Linear(hidden_size2, state_size),
        ).float().to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, prev_states, actions, dim=-1):
        inputs =  torch.cat((prev_states, actions), dim).float()
        pred_next_states = self.predictor(inputs)
        return pred_next_states

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, state_size, use_cuda, lr,
            hidden_size1=256, hidden_size2=64, chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
                nn.Linear(state_size, hidden_size1),
                nn.ReLU(),
                nn.Linear(hidden_size1, hidden_size2),
                nn.ReLU(),
                nn.Linear(hidden_size2, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))