from importlib.metadata import distribution
from re import I
from tkinter import W
import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim


from memory import Memory
from model import ActorNetwork, PredictorNetwork, CriticNetwork

class S3Agent(object):
    def __init__(self,
                state_size, 
                action_size, 
                hidden_size1, 
                hidden_size2,
                memory=None, 
                use_cuda=True,
                learning_rate=0.01,
                reward_decay=0.99,
                gae_lambda=0.95,
                policy_clip=0.2,
                batch_size=64,
                n_epochs=10,
                action_std_init=0.1
                ):
        # initialize hyperparameters / config
        self.input_size = state_size
        self.action_size = action_size
        self.reward_decay = reward_decay
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.action_std = action_std_init
 
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        # initialize memory and networks
        self.memory = memory if memory is None else Memory()
        self.memory.batch_size = batch_size
        
        self.predictor = PredictorNetwork(state_size, action_size,
            lr=learning_rate, use_cuda=use_cuda, hidden_size1=hidden_size1,
            hidden_size2=hidden_size2 ).to(self.device)

        self.actor = ActorNetwork(state_size, action_size, action_std_init=action_std_init,
            lr=learning_rate, use_cuda=use_cuda, hidden_size1=hidden_size1,
            hidden_size2=hidden_size2 ).to(self.device)

    def remember(self, curr_state, next_state, pred_state, action, reward, done):
        self.memory.add(curr_state, next_state, pred_state, action, reward, done)

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.actor.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)

    def evaluate(self, state, action):
        action_logprobs, dist_entropy = self.actor.calculate_entropy(state, action)
        pred_state = self.predictor(state)
        
        return action_logprobs, pred_state, dist_entropy

    def choose_action( self, state ): 
        action, action_logprob = self.actor.get_action(state)
        return action, action_logprob
    
    def predict(self, curr_state, action ):
        return self.predictor(curr_state, action)

    def forward(self, curr_state):
        action, action_logprob = self.actor( curr_state )
        pred_next_state = self.predict( curr_state, action )

        return action, pred_next_state

    def generate_rewards( self, states ):
        # generate predicted rewards
        predicted_rewards = []
        for state in states:
            predicted_rewards.append( self.calculate_reward( state ) )
        return torch.stack( predicted_rewards )

    def calculate_reward( self, state ):
        reward = self.memory.flat_get( state, "goal_lidar" )[0]
        return reward

    def calculate_rewards( self, state_flat ):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

    def generate_advantages( self, real_rewards, pred_rewards, dones ):
        # Monte Carlo estimate of returns
        advantages = np.zeros(len(real_rewards), dtype=np.float32)
        discounted_reward = 0
        discount = self.reward_decay * self.gae_lambda

        for t in range(len(advantages)-1, -1, -1):
            curr_reward = real_rewards[t]
            done = dones[t]
            pred_reward = pred_rewards[t]
            if done:
                discounted_reward = 0
            
            discounted_reward = curr_reward - pred_reward \
                                + (discount * discounted_reward)
            advantages[t] = discounted_reward
        
        return torch.tensor(advantages).to(self.device)
    
    def learn(self):
        real_rewards = self.generate_rewards( self.memory.next_states )
        for _ in range(self.n_epochs):
            curr_states_arr, next_states_arr, actions_arr,\
            old_logprobs_arr, rewards_arr, dones_arr, batches = \
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
                actor_loss = -self.generate_rewards( pred_states ).mean()
                predictor_loss = F.mse_loss(pred_states, next_states)

                total_loss = actor_loss + 0.5*predictor_loss
                self.actor.optimizer.zero_grad()
                self.predictor.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.predictor.optimizer.step()

        self.memory.clear_memory()

        return actor_loss, predictor_loss

# not yet working
class ActorCriticAgent:
       
    def __init__(self,
                state_size, 
                action_size, 
                hidden_size1, 
                hidden_size2,
                memory=None, 
                use_cuda=True,
                learning_rate=0.01,
                reward_decay=0.99,
                gae_lambda=0.95,
                policy_clip=0.2,
                batch_size=64,
                n_epochs=10,
                action_std_init=0.3
                ):
        # initialize hyperparameters / config
        self.input_size = state_size
        self.action_size = action_size
        self.reward_decay = reward_decay
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.action_std = action_std_init
 
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        # initialize memory and networks
        self.memory = memory if memory is None else Memory()
        self.memory.batch_size = batch_size

        self.actor = ActorNetwork(state_size, action_size, action_std_init=action_std_init,
            lr=learning_rate, use_cuda=use_cuda, hidden_size1=hidden_size1,
            hidden_size2=hidden_size2 ).to(self.device)

        self.critic = CriticNetwork(state_size,
            lr=learning_rate, use_cuda=use_cuda, hidden_size1=hidden_size1,
            hidden_size2=hidden_size2 ).to(self.device)

    def remember(self, curr_state, next_state, pred_state, action, reward, done):
        self.memory.add(curr_state, next_state, pred_state, action, reward, done)

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.actor.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)

    def choose_action( self, state ): 
        action, action_logprob = self.actor.get_action(state)
        return action, action_logprob

    def evaluate(self, state, old_action):
        dist = self.actor(state)
        
        # for single action continuous environments
        if self.action_size == 1:
            old_action = old_action.reshape(-1, self.action_size)

        action_logprobs = dist.log_prob(old_action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

    def forward(self, curr_state):
        action, action_logprob = self.actor( curr_state )
        pred_next_state = self.predict( curr_state, action )

        return action, pred_next_state

    def generate_rewards( self, states ):
        # generate predicted rewards
        predicted_rewards = []
        for state in states:
            predicted_rewards.append( self.calculate_reward( state ) )
        return torch.stack( predicted_rewards )

    def calculate_reward( self, state ):
        reward = self.memory.flat_get( state, "goal_lidar" )[0]
        return reward

    def calculate_rewards( self, state_flat ):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

    def generate_cumulative_rewards( self, real_rewards, dones ):
        # Monte Carlo estimate of returns
        rewards = np.zeros(len(real_rewards), dtype=np.float32)
        discounted_reward = 0
        discount = self.reward_decay * self.gae_lambda

        for t in range(len(rewards)-1, -1, -1):
            curr_reward = real_rewards[t]
            done = dones[t]
            if done:
                discounted_reward = 0

            discounted_reward = curr_reward + discount*discounted_reward
            rewards[t] = discounted_reward

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        return rewards
    
    def learn(self):
        # real_rewards = self.generate_rewards( self.memory.next_states )
        real_rewards = self.memory.rewards
        values_arr = self.generate_cumulative_rewards( real_rewards, self.memory.dones )

        for _ in range(self.n_epochs):
            curr_states_arr, next_states_arr, actions_arr,\
            old_logprobs_arr, rewards_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            for batch in batches:
                curr_states  = curr_states_arr[batch]
                actions = actions_arr[batch]
                values = values_arr[batch]

                new_logprobs, state_values, dist_entropy = self.evaluate( curr_states, actions )

                dists = self.actor(curr_states)
                critic_value = self.critic(curr_states)

                critic_value = torch.squeeze(critic_value)

                new_logprobs = dists.log_prob(actions)
                prob_ratio = torch.exp( new_logprobs - old_logprobs_arr[batch] )

                advantages = values - state_values.detach()
        
                weighted_probs = advantages * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantages
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                critic_loss = (values-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss + 0.01*dist_entropy.mean()

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

        return actor_loss, critic_loss