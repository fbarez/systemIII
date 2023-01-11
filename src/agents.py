""" Defines the models and learning methods.
"""
import time
from typing import Optional
import numpy as np
import torch

from params import Params
from memory import Memory
from model import ActorNetwork, PredictorNetwork, CriticNetwork, PenaltyModel
from learn import learn

class Agent:
    """ Define generic Agent model for RL learning """
    def __init__(self, params:Params, memory:Optional[Memory]=None):
        # initialize hyperparameters / config
        self.params = params
        self.device = torch.device('cuda' if self.params.use_cuda else 'cpu')

        # initialize memory and networks
        self.memory = Memory( self.params ) if (memory is None) else memory

        # shortcut parameters
        self.gae_lambda   = self.params.gae_lambda
        self.reward_decay = self.params.reward_decay
        self.batch_size   = self.params.batch_size
        self.action_std   = self.params.action_std

        self.actor : Optional[ActorNetwork] = None
        self.predictor : Optional[PredictorNetwork] = None
        self.value_critic : Optional[CriticNetwork] = None
        self.cost_critic : Optional[CriticNetwork] = None
        self.penalty : Optional[PenaltyModel] = None

        self.models = []

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

class S3Agent(Agent):
    """ Define the special System 3 Agent class. """
    def __init__(self, params:Params, memory:Optional[Memory]=None):
        super(S3Agent, self).__init__(params, memory)

        self.name = "s3"
        self.actor     = ActorNetwork( params )
        self.predictor = PredictorNetwork( params )
        self.value_critic = CriticNetwork( params, "value_critic" )

        self.models = [ self.actor, self.predictor, self.value_critic ]

        if params.train_cost_critic:
            self.cost_critic = CriticNetwork( params, "cost_critic" )
            self.models.append( self.cost_critic )

        self.penalty = PenaltyModel( params )

        self.params.clipped_advantage = True

    def calculate_constraint( self, index, state, memory ):
        raise NotImplementedError

    def calculate_all_constraints(self, states):
        constraints = torch.zeros(len(states), dtype=torch.float32).to(self.device)
        for i, state in enumerate(states):
            constraints[i] = self.calculate_constraint(i, state, self.memory)
        return constraints

    def learn(self):
        return learn(self)

class ActorCriticAgent( Agent ):
    """Defines the standard Actor Critic PPO agent"""
    def __init__(self, params:Params, memory:Optional[Memory]=None):
        super(ActorCriticAgent, self).__init__(params, memory)

        self.name = "ac"
        self.actor  = ActorNetwork( params )
        self.value_critic = CriticNetwork( params, "value_critic" )

        self.models = [ self.actor, self.value_critic ]

        if params.train_cost_critic:
            self.cost_critic = CriticNetwork( params, "cost_critic" )
            self.models.append( self.cost_critic )

        self.penalty = PenaltyModel( params )

        self.params.clipped_advantage = True

    def learn(self):
        return learn(self)
