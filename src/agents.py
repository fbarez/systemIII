from re import I
from tkinter import W
import numpy as np

import time
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
from typing import Optional


from memory import Memory
from model import ActorNetwork, PredictorNetwork, CriticNetwork
from params import Params

class Agent:
    def __init__(self, params:Params, memory:Optional[Memory]=None):
        # initialize hyperparameters / config
        self.params = params    
        self.device = torch.device('cuda' if self.params.use_cuda else 'cpu')
        
        # initialize memory and networks
        self.memory = Memory( self.params.use_cuda ) if memory is None else memory

        # shortcut parameters
        self.gae_lambda   = self.params.gae_lambda
        self.reward_decay = self.params.reward_decay
        self.batch_size   = self.params.batch_size
        self.action_std   = self.params.action_std

        self.models = []

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
        action, action_logprob, action_mean = self.actor.get_action(state, training=training)
        return action, action_logprob, action_mean

    def calculate_cumulative_rewards( self, rewards=None, dones=None ):
        rewards = self.memory.rewards if rewards is None else rewards
        dones   = self.memory.dones   if   dones is None else dones

        cumulative_rewards = np.zeros(len(rewards), dtype=np.float32)
        cumulative_rewards[-1] = rewards[-1]
        for t in range( len(rewards)-2, -1, -1 ):
            cumulative_rewards[t] += rewards[t] 
            cumulative_rewards[t] += self.reward_decay*self.gae_lambda*cumulative_rewards[t+1]
            #if dones[t]:
            #    cumulative_rewards[t] = rewards[t]
        cumulative_rewards = torch.tensor(cumulative_rewards).to(self.device)

        return cumulative_rewards
    
    def calculate_cumulative_values(self, values=None, dones=None):
        values = self.memory.values if values is None else values
        dones   = self.memory.dones if  dones is None else dones

        cumulative_values = np.zeros(len(values), dtype=np.float32)
        cumulative_values[-1] = values[-1]
        for t in range( len(values)-2, -1, -1 ):
            cumulative_values[t] += values[t]
            cumulative_values[t] += self.reward_decay*self.gae_lambda*cumulative_values[t+1]
            if not dones[t]:
                cumulative_values[t] -= self.reward_decay*values[t+1]
        cumulative_values = torch.tensor(cumulative_values).to(self.device)

        return cumulative_values

    def generate_advantages( self, rewards=None, values=None, dones=None ):
        cumulative_rewards = self.calculate_cumulative_rewards()
        cumulative_values  = self.calculate_cumulative_values()
        
        advantages = cumulative_rewards - cumulative_values

        return advantages
    
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
    def __init__(self, params:Params, memory:Optional[Memory]=None):
        super(S3Agent, self).__init__(params, memory)

        self.name = "s3"
        self.actor     = ActorNetwork( params )
        self.predictor = PredictorNetwork( params )
        self.critic    = CriticNetwork( params )

        self.models = [ self.actor, self.predictor, self.critic ]

    def calculate_constraint( self, index, state, memory ):
        return 1

    def calculate_all_constraints(self, states):
        constraints = torch.zeros(len(states), dtype=torch.float32).to(self.device)
        for i, state in enumerate(states):
            constraints[i] = self.calculate_constraint(i, state, self.memory)
        return constraints

    def calculate_constrained_rewards( self,
            cumulative_rewards:torch.Tensor,
            constraints_arr:torch.Tensor
            ):
        # constrained_rewards = torch.min( cumulative_rewards, cumulative_rewards*constraints_arr )
        # code for multiplying constraints by cumulative reward
        delta_arr = torch.zeros_like( cumulative_rewards )
        constrained_rewards = torch.zeros_like( cumulative_rewards )
        constrained_rewards[-1] = cumulative_rewards[-1] * constraints_arr[-1]
        for i in range(len(cumulative_rewards)-2, -1, -1):
            reward_diff = cumulative_rewards[i] * ( 1 - constraints_arr[i] )
            options = [ torch.tensor(0), reward_diff, self.reward_decay*delta_arr[i+1] ]
            delta_arr[i] = torch.max( torch.stack( options ) )
            constrained_rewards[i] = cumulative_rewards[i] - delta_arr[i]
        del delta_arr
        return constrained_rewards

    def learn(self):
        # prepare advantages and other tensors used for training
        memory = self.memory.prepare()
        cumulative_rewards = self.calculate_cumulative_rewards()
        cumulative_values  = self.calculate_cumulative_values()
        constraints_arr = self.calculate_all_constraints(self.memory.next_states)
        constrained_rewards = self.calculate_constrained_rewards( 
            cumulative_rewards=cumulative_rewards, constraints_arr=constraints_arr )
        advantages_arr = constrained_rewards - cumulative_values 

        # begin training loops
        for epoch in range(self.params.n_epochs):
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
                constraints  = constraints_arr[batch]

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
                critic_value = self.critic(pred_states).squeeze()

                # calculate actor loss
                prob_ratio = ( new_logprobs - old_logprobs ).exp()
                clip = self.params.policy_clip
                weighted_probs = advantages * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-clip,1+clip)*advantages
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                # calculate critic loss
                returns = advantages + values.squeeze()
                returns = torch.min( returns, constraints * returns )
                critic_loss = torch.nn.HuberLoss("mean")(returns, critic_value)

                # backprop the loss
                total_loss = actor_loss + 0.5*critic_loss - 0.01*entropies.mean() 
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                self.predictor.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

            # calculate KL divergence to check for early stopping
            do_early_stop, kl = self.check_kl_early_stop()
            if do_early_stop:
                print("Early stopping at epoch {} with KL divergence {}".format(epoch, kl))
                break
        
        self.memory.clear_memory() 

        losses = { 'actor': actor_loss, "predictor": predictor_loss, 'critic': critic_loss }
        return losses

# not yet working
class ActorCriticAgent( Agent ):
       
    def __init__(self, params:Params, memory:Optional[Memory]=None):
        super(ActorCriticAgent, self).__init__(params, memory)

        self.name = "ac"
        self.actor  = ActorNetwork( params )
        self.critic = CriticNetwork( params )
        
        self.models = [ self.actor, self.critic ]
 
    def learn(self):
        # prepare advantages and other tensors used for training
        memory = self.memory.prepare()
        advantages_arr = self.generate_advantages()

        # begin training loops
        for epoch in range(self.params.n_epochs):
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

            # calculate KL divergence to check for early stopping
            do_early_stop, kl = self.check_kl_early_stop()
            if do_early_stop:
                print("Early stopping at epoch {} with KL divergence {}".format(epoch, kl))
                break

        self.memory.clear_memory() 

        losses = { 'actor': actor_loss, 'critic': critic_loss }
        return losses
