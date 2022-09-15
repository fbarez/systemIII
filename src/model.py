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
    def __init__(self, state, hidden_size1, hidden_size2, next_state, use_cuda=True):
        """[summary]

        Args:
            state ([type]): [The state dimentions]
            hidden_size1 ([type]): [size  of hidden layers]
            hidden_size2 ([type]): [size of hidden layers]
            next_state ([type]): [values of the next state - dimention should be = to state]
            use_cuda (bool, optional): [description]. Defaults to True.
        """        
        super(S3Model, self).__init__()
        
        self.state = state
        self.next_state = next_state

        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.forward_net = nn.Sequential(
            
            nn.Linear(input, hidden_size1),
            nn.LeakyReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size2, next_state)
        )

        # for p in self.modules():
        #     if isinstance(p, nn.Conv2d):
        #         init.kaiming_uniform_(p.weight)
        #         p.bias.data.zero_()

        #     if isinstance(p, nn.Linear):
        #         init.kaiming_uniform_(p.weight, a=1.0)
        #         p.bias.data.zero_()

    def forward(self, inputs):
        state, next_state, action = inputs
        #phi1 = encode_state
        #phi2 = encoded_next_state
        #encode_state = self.feature(state)
        #encode_next_state = self.feature(next_state)
        # get pred action
        # pred_action = torch.cat((state, next_state), 1) #encode should inputs be images
        # pred_action = self.inverse_net(pred_action)
        # # ---------------------

        # get pred next state
        #pred_next_state = torch.cat((state, action), 1)
        #pred_next_state = self.forward_net(pred_next_state)

        # # residual
        # for i in range(4):
        #     pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, action), 1))
        #     pred_next_state_feature_orig = self.residual[i * 2 + 1](
        #         torch.cat((pred_next_state_feature, action), 1)) + pred_next_state_feature_orig

        pred_next_state = self.forward_net(torch.cat((state, action), 1))

        real_next_state = next_state
        #it should return st, st+1, pred action
        return real_next_state, pred_next_state #, pred_action
