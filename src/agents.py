from importlib.metadata import distribution
from re import I
from tkinter import W
import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
from typing import Optional


from memory import Memory
from model import ActorNetwork, PredictorNetwork, CriticNetwork
from params import Params

class Agent:
    def __init__(self, params:Params):
        # initialize hyperparameters / config
        self.params = params    
        self.device = torch.device('cuda' if self.params.use_cuda else 'cpu')

        # initialize memory and networks
        self.memory = Memory( self.params.use_cuda )

        # shortcut parameters
        self.gae_lambda   = self.params.gae_lambda
        self.reward_decay = self.params.reward_decay
        self.batch_size   = self.params.batch_size
        self.action_std   = self.params.action_std

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

    def choose_action( self, state ): 
        if not hasattr(self, 'actor'):
            raise Exception("Agent has no attribute 'actor'")
        action, action_logprob = self.actor.get_action(state)
        return action, action_logprob

    def generate_advantages( self, rewards=None, values=None, dones=None ):
        rewards = self.memory.rewards if rewards is None else rewards
        values  = self.memory.values  if  values is None else values
        dones   = self.memory.dones   if   dones is None else dones
        
        advantages = np.zeros(len(rewards), dtype=np.float32)
        advantages[-1] = rewards[-1] - values[-1]
        for t in range( len(rewards)-2, -1, -1 ):
            advantages[t] += rewards[t] - values[t] 
            advantages[t] += self.reward_decay*self.gae_lambda*advantages[t+1]
            if not dones[t]:
                advantages[t] += self.reward_decay*values[t+1]
        advantages = torch.tensor(advantages).to(self.device)

        return advantages
    
    def generate_batches(self):
        n_states = len(self.memory.curr_states) 
        batch_start = np.arange(0, n_states, self.batch_size) 
        indices = np.arange(n_states, dtype=np.int64) 
        np.random.shuffle(indices) 
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return batches

class S3Agent(Agent):
    def __init__(self, params:Params):
        super(S3Agent, self).__init__(params)

        self.actor     = ActorNetwork( params )
        self.predictor = PredictorNetwork( params )
        self.critic    = CriticNetwork( params )
    
    def learn(self):
        # prepare advantages and other tensors used for training
        memory = self.memory.prepare()
        advantages_arr = self.generate_advantages()
        advantages_arr = advantages_arr - memory.rewards

        # begin training loops
        for _ in range(self.params.n_epochs):
            batches = self.generate_batches()

            for batch in batches:
                # get required info from batches
                curr_states  = memory.curr_states[batch]
                next_states  = memory.next_states[batch]
                old_logprobs = memory.logprobs[batch]
                actions      = memory.actions[batch]
                rewards      = memory.rewards[batch]
                values       = memory.values[batch]
                advantages   = advantages_arr[batch]

                # train in two separate steps.
                # Train the predictor
                pred_states = self.predictor(curr_states, actions)
                predictor_loss = torch.nn.HuberLoss("mean")(next_states, pred_states)
                self.predictor.optimizer.zero_grad()
                predictor_loss.backward()
                self.predictor.optimizer.step()

                # Train the actor and critic
                # run the models
                new_logprobs, entropies = self.actor.calculate_entropy(curr_states, actions)
                pred_states  = self.predictor(curr_states, actions)
                critic_value = self.critic(pred_states)

                # calculate actor loss
                prob_ratio = ( new_logprobs - old_logprobs ).exp()
                clip = self.params.policy_clip
                weighted_probs = advantages * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-clip,1+clip)*advantages
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                # calculate critic loss
                returns = advantages + values - rewards
                critic_loss = torch.nn.HuberLoss("mean")(returns, critic_value)

                # backprop the loss
                total_loss = actor_loss + 0.5*critic_loss - 0.01*entropies.mean() 
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                self.predictor.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory() 

        losses = { 'actor': actor_loss, 'critic': critic_loss, "predictor": predictor_loss }
        return losses

# not yet working
class ActorCriticAgent( Agent ):
       
    def __init__(self, params:Params):
        super(ActorCriticAgent, self).__init__(params)

        self.actor  = ActorNetwork( params )
        self.critic = CriticNetwork( params )
 
    def learn(self):
        # prepare advantages and other tensors used for training
        memory = self.memory.prepare()
        advantages_arr = self.generate_advantages()

        # begin training loops
        for _ in range(self.params.n_epochs):
            batches = self.generate_batches()

            for batch in batches:
                # get required info from batches
                states       = memory.curr_states[batch]
                old_logprobs = memory.logprobs[batch]
                actions      = memory.actions[batch]
                values       = memory.values[batch]
                advantages   = advantages_arr[batch]

                # run the models
                new_logprobs, entropies = self.actor.calculate_entropy(states, actions)
                critic_value = self.critic(states)

                # calculate actor loss
                prob_ratio = ( new_logprobs - old_logprobs ).exp()
                clip = self.params.policy_clip
                weighted_probs = advantages * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-clip,1+clip)*advantages
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                # calculate critic loss
                returns = advantages + values
                critic_loss = torch.nn.HuberLoss()(returns, critic_value)
                critic_loss = critic_loss.mean()

                # backprop the loss
                total_loss = actor_loss + 0.5*critic_loss - 0.01*entropies.mean() 
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory() 

        losses = { 'actor': actor_loss, 'critic': critic_loss }
        return losses
