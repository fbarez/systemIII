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

class S3Agent(object):
    def __init__(self, params:Params):
        super(S3Agent, self).__init__(params)

        self.actor     = ActorNetwork( params )
        self.predictor = PredictorNetwork( params )
        self.critic    = CriticNetwork( params )

    def evaluate(self, state, action):
        action_logprobs, dist_entropy = self.actor.calculate_entropy(state, action)
        pred_state = self.predictor(state)
        
        return action_logprobs, pred_state, dist_entropy
    
    def learn(self):
        real_rewards = self.generate_rewards( self.memory.next_states )
        for _ in range(self.n_epochs):
            curr_states_arr, next_states_arr, pred_states_arr, actions_arr,\
            old_logprobs_arr, rewards_arr, values_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            pred_states_arr = self.predictor( curr_states_arr, actions_arr )
            pred_rewards = self.generate_rewards( pred_states_arr )
            advantages = self.generate_advantages( real_rewards, pred_rewards, dones_arr )

            for batch in batches:
                curr_states  = curr_states_arr[batch]
                next_states  = next_states_arr[batch]
                actions = actions_arr[batch]

                dists = self.actor(curr_states)
                pred_states = self.predictor(curr_states, actions)

                new_logprobs = dists.log_prob(actions)
                prob_ratio = torch.exp( new_logprobs - old_logprobs_arr[batch] )
                """
                weighted_probs = advantages[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantages[batch]

                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).sum()
                """ 
                actor_loss     = -self.generate_rewards( pred_states ).mean()
                predictor_loss = F.mse_loss(pred_states, next_states)

                self.actor.optimizer.zero_grad()
                actor_loss.backward( retain_graph=True )
                self.actor.optimizer.step()

                self.predictor.optimizer.zero_grad()
                predictor_loss.backward()
                self.predictor.optimizer.step()

        self.memory.clear_memory()
        losses = { 'actor': actor_loss, 'predictor': predictor_loss }

        return actor_loss, losses

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
