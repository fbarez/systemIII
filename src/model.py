""" Define the Neural Networks and Models used by the Agent Models
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.distributions import MultivariateNormal, Categorical

from params import Params

# for some reason, the following do not work be default
# pylint: disable=function-redefined, abstract-method, missing-class-docstring
class Categorical(Categorical):
    @property
    def mode(self):
        return self.probs.argmax(dim=-1)

# pylint: disable=function-redefined, abstract-method, missing-class-docstring
class MultivariateNormal(MultivariateNormal):
    @property
    def mode(self):
        return self.loc


class ActorNetwork(nn.Module):
    def __init__(self, params:Params):
        super(ActorNetwork, self).__init__()

        # shortcut to parameters
        p = params
        self.device = torch.device('cuda' if p.use_cuda else 'cpu')

        self.update_checkpoint(p)
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
            ).float().to(self.device)
        else:
            self.actor = nn.Sequential(
                nn.Linear(p.state_size, p.hidden_size1),
                nn.ReLU(),
                nn.Linear(p.hidden_size1, p.hidden_size2),
                nn.ReLU(),
                nn.Linear(p.hidden_size2, p.action_size),
                nn.Softmax(dim=-1)
            ).float().to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=p.learning_rate)
        self.to(self.device)

    def cov_mat(self, action_mean):
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        return cov_mat

    # pylint: disable=not-callable
    def forward(self, state):
        if self.actions_continuous:
            action_mean = self.actor(state)
            cov_mat = self.cov_mat(action_mean)
            distribution = MultivariateNormal(action_mean, cov_mat)

        else:
            action_probs = self.actor(state)
            distribution = Categorical(action_probs)

        return distribution

    def get_action(self, state, training=True, detach=True):
        if detach:
            with torch.no_grad():
                distribution = self.forward(state)
        else:
            distribution = self.forward(state)

        if training:
            action = distribution.sample()
        else:
            action = distribution.mode

        action_logprob = distribution.log_prob(action)
        action_mean = distribution.mode

        return action, action_logprob, action_mean

    def calculate_entropy(self, state, action):
        distribution = self.forward(state)

        action_logprobs = distribution.log_prob(action)
        dist_entropy = distribution.entropy()

        return action_logprobs, dist_entropy

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.action_var = \
            torch.full((self.action_size,), new_action_std**2 ).to(self.device)

    def update_checkpoint(self, params:Params):
        self.checkpoint_file = os.path.join(
            params.checkpoint_dir, params.agent_type+'_actor_'+params.instance_name
        )

    def calculate_kl_divergence(self, states, old_means):
        with torch.no_grad():
            new_means = self.forward(states).loc
        kl  = 0.5 * ( (new_means - old_means)**2 ).sum(axis=-1).mean()
        kl /= (self.action_std**2)
        return kl

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class PredictorNetwork(nn.Module):
    def __init__(self, params:Params):
        super(PredictorNetwork, self).__init__()
        p = params
        self.device = torch.device('cuda' if p.use_cuda else 'cpu')

        self.update_checkpoint(p)
        self.predictor = nn.Sequential(
                nn.Linear(p.state_size+p.action_size, p.hidden_size1),
                nn.LeakyReLU(),
                nn.Linear(p.hidden_size1, p.hidden_size2),
                nn.LeakyReLU(),
                nn.Linear(p.hidden_size2, p.state_size),
        ).float().to(self.device)

        self.action_size = p.action_size
        self.optimizer = optim.Adam(self.parameters(), lr=p.learning_rate)
        self.to(self.device)

    # pylint: disable=not-callable
    def forward(self, state, action, dim=-1):
        if actions.dtype == torch.int64:
            actions = torch.nn.functional.one_hot(action,
                num_classes=self.action_size).float()

        # residual predictor
        inputs =  torch.cat((state, actions), dim).float()
        pred_next_state = self.predictor(inputs) + state
        return pred_next_state

    def update_checkpoint(self, params:Params):
        self.checkpoint_file = os.path.join(
            params.checkpoint_dir, params.agent_type+'_predictor_'+params.instance_name
        )

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, params:Params, name='critic'):
        super(CriticNetwork, self).__init__()
        p = params
        self.name = name
        self.device = torch.device('cuda' if p.use_cuda else 'cpu')

        self.update_checkpoint(p)
        self.critic = nn.Sequential(
                nn.Linear(p.state_size, p.hidden_size1),
                nn.ReLU(),
                nn.Linear(p.hidden_size1, p.hidden_size2),
                nn.ReLU(),
                nn.Linear(p.hidden_size2, 1)
        ).float().to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=p.learning_rate)
        self.to(self.device)

    def forward(self, state):
        # pylint: disable=not-callable
        value = self.critic(state)
        return value

    def update_checkpoint(self, params:Params):
        self.checkpoint_file = os.path.join(
            params.checkpoint_dir,
            params.agent_type + f"_{self.name}_" + params.instance_name
        )

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class PenaltyModel:
    """ Pytorch Alternative to penalty learning done in safety-starter-agents.
    See original implementation here:
    https://github.com/openai/safety-starter-agents/blob/master/safe_rl/pg/run_agent.py#L134
    """
    def __init__(self, params:Params):
        self.cost_limit = params.cost_limit # default 25
        self.penalty_init = params.penalty_init # default 1.
        self.penalty_lr = params.penalty_lr # default 5e-2
        self.learn_penalty = params.learn_penalty # depends on if penalty is used
        self.penalty_param_loss = params.penalty_param_loss # default True

        # initialize penalty parameter
        penalty_param_init = np.log( max( np.exp(self.penalty_init)-1, 1e-8 ) )
        self.penalty_param = torch.nn.param(penalty_param_init, dtype=torch.float32)

        # initialize adam optimizer
        self.optimizer = torch.optim.Adam(self.penalty_param, lr=self.penalty_lr)

    def calculate_penalty(self):
        """Calculates the penalty based on the penalty parameters"""
        # calculate penalty based on parameter
        penalty = torch.nn.softplus(self.penalty_param)
        return penalty

    def use_penalty(self):
        """Returns penalty but without training graphs. i.e: torch.nograd()"""
        with torch.no_grad():
            return self.calculate_penalty()

    def learn(self, episode_costs: Tensor):
        # learn if needed
        if self.learn_penalty:
            if self.penalty_param_loss:
                penalty_loss = - self.penalty_param * (episode_costs-self.cost_limit)

            else:
                penalty = self.calculate_penalty()
                penalty_loss = - penalty * (episode_costs - self.cost_limit)

            self.optimizer.zero_grad()
            penalty_loss.backward()
            self.optimizer.step()

        return self
