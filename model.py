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
                learning_rate=0.01,
                state_mapping={}
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
        ).float().to(self.device)

        self.predictor = nn.Sequential( 
            nn.Linear(state_size+action_size, hidden_size1),
            nn.LeakyReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size2, state_size)
        ).float().to(self.device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.optimizer_predictor = optim.Adam(self.predictor.parameters(), lr=learning_rate)

        self.state_mapping = state_mapping
    
    def flat_get(self, current_state_flat, key):
        if not self.state_mapping:
            raise Exception("mapping not defined")

        return current_state_flat[self.state_mapping[key][0]:self.state_mapping[key][1]]
    
    def choose_action(self, state):
        state = torch.tensor(state).to(self.device)
        action = self.actor(state.float())

        return action

    def train_predictor( self, prev_states, actions, next_states ):
        self.optimizer_predictor.zero_grad()
        inputs =  torch.cat((prev_states, actions), 1).float()
        pred_next_states = self.predictor(inputs)
        next_states = next_states.float()
        loss = F.mse_loss(pred_next_states, next_states)
        loss.backward()
        self.optimizer_predictor.step()

        return loss

    def calculate_reward( self, state_flat ):
        reward = self.flat_get(state_flat, "goal_lidar")[0]
        return reward

    def train_actor( self, prev_states ):
        self.optimizer_actor.zero_grad()
        action_prob = self.actor(prev_states.float())
        inputs = torch.cat((prev_states, action_prob), 1).float()
        pred_next_states = self.predictor(inputs)
        loss = torch.stack([ -self.calculate_reward(s) for s in pred_next_states ]).sum()
        loss.backward()
        self.optimizer_actor.step()

        return loss.sum()

    def forward(self, curr_state):
        action = self.get_action( curr_state )
        pred_next_state = self.predictor(torch.cat((curr_state, action), 1))

        return action, pred_next_state
    
    def train(self, prev_states, actions, next_states):
        predictor_loss = self.train_predictor( prev_states, actions, next_states )
        actor_loss = self.train_actor( prev_states )
        return predictor_loss, actor_loss
