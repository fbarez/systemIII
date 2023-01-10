import numpy as np

import time
import torch
from typing import Optional


from memory import Memory
from model import ActorNetwork, PredictorNetwork, CriticNetwork, PenaltyModel
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

        self.actor : Optional[ActorNetwork] = None
        self.predictor : Optional[PredictorNetwork] = None
        self.value_critic : Optional[CriticNetwork] = None
        self.cost_critic : Optional[CriticNetwork] = None
        self.penalty : Optional[PenaltyModel] = None

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
 
def learn(agent: Agent):
    # prepare advantages and other tensors used for training
    agent.memory.calculate_advantages()
    memory = agent.memory.prepare()
    kl_target_reached = False

    # begin training loops
    for epoch in range(agent.params.n_epochs):

        # Fine-tune the penalty parameter
        agent.penalty.learn( memory.episode_costs )
        curr_penalty = agent.penalty.get_penalty()

        # Create the batches for training agent networks
        batches = agent.generate_batches()
        for batch in batches:
            # get required info from batches
            curr_states  = memory.curr_states[batch]
            next_states  = memory.next_states[batch]
            old_logprobs = memory.logprobs[batch]
            actions      = memory.actions[batch]

            returns         = memory.returns[batch]
            cost_returns    = memory.cost_returns[batch]
            advantages      = memory.advantages[batch]
            cost_advantages = memory.cost_advantages[batch]

            # Train in two separate steps:
            # 1. Train the Predictor
            # 2. Train the Actor and Critic(s) together

            # initialize test variables
            has_predictor = hasattr(agent, 'predictor')
            has_cost_critic = hasattr(agent, 'cost_critic')

            # 1. Train the predictor (if the agent has one)
            if hasattr(agent, 'predictor'):
                # run predictor
                pred_states = agent.predictor(curr_states, actions)

                # Get loss
                loss_fn = torch.nn.MSELoss() # torch.nn.HuberLoss("mean")
                predictor_loss = loss_fn(next_states, pred_states)

                # Update predictor
                agent.predictor.optimizer.zero_grad()
                predictor_loss.backward()
                agent.predictor.optimizer.step()

            # 2. Train the models for the actor and critic(s)
            # 2.1 Run all the models to build gradients:
            # a) Actor
            new_logprobs, entropies = agent.actor.calculate_entropy(curr_states, actions)
            
            # b) Run predictor, or just use current states
            states = curr_states
            if has_predictor:
                states = agent.predictor(curr_states, actions)
            
            # c) Run value critic
            value_critic_value = agent.value_critic(states).squeeze()

            # d) Run cost critic, if needed
            if has_cost_critic:
                cost_critic_value   = agent.cost_critic(states).squeeze()

            # 2.2 Calculate KL divergence of Actor policy.
            if not kl_target_reached:
                do_early_stop, kl = agent.check_kl_early_stop()
                if do_early_stop:   
                    print(f"Early stopping at epoch {epoch} with KL divergence {kl}")
                kl_target_reached = True

            # 2.3 Calculate actor loss, only if KL divergence is low enough
            actor_loss = 0
            if not kl_target_reached:
                # calculate scaled advantages
                prob_ratio = ( new_logprobs - old_logprobs ).exp() # Likelihood ratio
                clip = agent.params.policy_clip
                surrogate_advantages = advantages * prob_ratio # scaled advantage

                # if using PPO, calculate scaled + clipped advantages
                if agent.params.clipped_advantage:
                    scaled_clipped_advantages = \
                        torch.clamp(prob_ratio, 1-clip, 1+clip) * advantages
                    surrogate_advantages = \
                        torch.min(surrogate_advantages, scaled_clipped_advantages)
                
                actor_objective = surrogate_advantages.mean()

                # Sometimes use cost advantages. See safety-starter-agents
                if has_cost_critic:
                    surrogate_cost = (prob_ratio * cost_advantages).mean()
                    actor_objective -= curr_penalty * surrogate_cost
                    actor_objective /= (1 + curr_penalty)

                # Loss for actor policy is negative objective
                actor_loss = - actor_objective

            # 2.4 Calculate loss for value critic
            value_critic_loss = torch.mean((returns - value_critic_value)**2)
            critic_loss = value_critic_loss

            # 2.5 Calculate loss for cost critic, if needed
            if has_cost_critic:
                cost_critic_loss  = torch.mean((cost_returns - cost_critic_value)**2)
                critic_loss += cost_critic_loss

            # 2.6 Add optional entropy loss term
            entropy_regularization = agent.params.entropy_regularization
            if entropy_regularization != 0:
                entropy_loss += entropy_regularization * entropies.mean()

            # 2.7 Calculate total loss for actor and critics
            total_loss = actor_loss + 0.5*critic_loss + entropy_loss
            for model in agent.models:
                model.optimizer.zero_grad()

            # 2.8 Backprop loss for actor and critic(s)
            total_loss.backward()
            agent.actor.optimizer.step()
            agent.value_critic.optimizer.step()
            agent.cost_critic.step() if (not agent.reward_penalized) else None
 
    agent.memory.clear_memory() 

    losses = { 'actor': actor_loss, "predictor": predictor_loss, 'critic': critic_loss }
    return losses
