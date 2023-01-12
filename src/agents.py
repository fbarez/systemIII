""" Defines the models and learning methods.
"""
from typing import Optional
import torch

from params import Params
from memory import Memory
from model import ActorNetwork, PredictorNetwork, CriticNetwork, PenaltyModel
from agent_base import Agent
from learn import learn

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
            self.models.append( self.penalty )

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
            self.models.append( self.penalty)

        self.params.clipped_advantage = True

    def learn(self):
        return learn(self)
