import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        """[summary]

        Args:
            input_size ([type]): [This is the state values]
            hidden_size1 ([type]): [number of hidden layers]
            hidden_size2 ([type]): [number of hidden layers]
            output_size ([type]): [this is the action by the actor and scalr value by the critic stating how 'good' the action]
        """        
        super(ActorCriticNetwork, self).__init__()

        self.critic = nn.Sequential(
                nn.Linear(input_size, hidden_size1),
                nn.ReLU(),
                nn.Linear(hidden_size1, hidden_size2),
                nn.ReLU(),
                nn.Linear(hidden_size2, 1)
        )            
        self.actor = nn.Sequential(
                nn.Linear(input_size, hidden_size1),
                nn.ReLU(),
                nn.Linear(hidden_size1, hidden_size2),
                nn.ReLU(),
                nn.Linear(hidden_size2, output_size),
                nn.Softmax(dim=-1)
        )


    def forward(self, state):
        x = self.feature(state)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value


class S3Model(nn.Module):
    def __init__(self,
                state_size, 
                action_size, 
                hidden_size1, 
                hidden_size2, 
                use_cuda=True,
                learning_rate=0.01
                ):
        """[summary]

        Args:
            state ([type]): [The state dimentions]
            hidden_size1 ([type]): [size  of hidden layers]
            hidden_size2 ([type]): [size of hidden layers]
            next_state ([type]): [values of the next state - dimention should be = to state]
            use_cuda (bool, optional): [description]. Defaults to True.
        """        
        super(S3Model, self).__init__()
        
        self.state_size  = state_size
        self.action_size = action_size

        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, action_size),
            nn.Softmax(dim=-1)
        )

        self.forward_net = nn.Sequential( 
            nn.Linear(state_size+action_size, hidden_size1),
            nn.LeakyReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size2, state_size)
        )

        self.optimizer_actor = optim.Adam(self.actor, lr=learning_rate)
        self.optimizer_ff    = optim.Adam(self.forward_net, lr=learning_rate)

    def get_action(self, state):
        state = torch.Tensor(state).to(self.device)
        action_prob = state.float()
        print( action_prob )

        action = self.random_choice_prob_index(action_prob)

        return action, action_prob

    def forward(self, curr_state):
        action, action_prob = self.get_action( curr_state )
        pred_next_state = self.forward_net(torch.cat((curr_state, action), 1))

        return action, pred_next_state
    
    def train(self):
        return None
