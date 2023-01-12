""" Define an base Agent class from which to instantiate other agents
"""
import time
from typing import Optional
import numpy as np
import torch

from params import Params
from memory import Memory
from model import ActorNetwork, PredictorNetwork, CriticNetwork, PenaltyModel

class Agent:
    """ Define generic Agent model for RL learning """
    def __init__(self, params:Params, memory:Optional[Memory]=None):
        # initialize hyperparameters / config
        self.params = params
        self.device = torch.device('cuda' if self.params.use_cuda else 'cpu')

        # initialize memory and networks
        self.memory = Memory( self.params ) if (memory is None) else memory

        self.predictor    : Optional[PredictorNetwork] = None
        self.actor        : Optional[ActorNetwork]     = None
        self.value_critic : Optional[CriticNetwork]    = None
        self.cost_critic  : Optional[CriticNetwork]    = None
        self.penalty      : Optional[PenaltyModel]     = None

        # shortcut parameters
        self.gae_lambda   = self.params.gae_lambda
        self.reward_decay = self.params.reward_decay
        self.batch_size   = self.params.batch_size
        self.action_std   = self.params.action_std

        self.models = []

    @property
    def has_actor(self):
        return not (self.actor is None)

    @property
    def has_predictor(self):
        return not (self.predictor is None)

    @property
    def has_value_critic(self):
        return not (self.value_critic is None)

    @property
    def has_cost_critic(self):
        return not (self.cost_critic is None)

    @property
    def has_penalty(self):
        return not (self.penalty is None)

    def run_if_has(self, attr:str, **kwargs):
        if not hasattr(self, attr):
            return torch.tensor(0)
        agent_method = getattr(self, attr)
        return agent_method(**kwargs)

    def set_action_std(self, new_action_std):
        if not hasattr(self, 'actor'):
            raise Exception("Agent has no attribute 'actor'")
        self.action_std = new_action_std
        self.actor.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)

    def choose_action( self, state, training=True ):
        if not hasattr(self, 'actor'):
            raise Exception("Agent has no attribute 'actor'")
        action, action_logprob, action_mean = \
            self.actor.get_action(state, training=training)
        return action, action_logprob, action_mean

    def generate_batches(self):
        n_states = len(self.memory.curr_states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return batches

    def check_kl_early_stop(self, memory:Optional[Memory]=None):
        if not self.params.kl_target:
            return False, None
        if not hasattr(self, 'actor'):
            raise Exception("Agent has no attribute 'actor'")

        memory = self.memory if memory is None else memory
        curr_states  = memory.curr_states
        action_means = memory.action_means

        kl = self.actor.calculate_kl_divergence( curr_states, action_means )
        if kl > self.params.kl_target:
            return True, kl

        return False, kl

    def save_models(self):
        time_str = time.strftime("%Y.%m.%d.%H:%M:%S", time.localtime())
        self.params.instance_name = time_str

        for model in self.models:
            model.update_checkpoint(self.params)
            model.save_checkpoint()

    def load_models(self):
        for model in self.models:
            model.load_checkpoint()

    def learn(self):
        raise NotImplementedError