""" Define learning functions for training the model
"""

import time
import torch
import numpy as np
from agent_base import Agent

def learn(agent: Agent):
    # prepare advantages and other tensors used for training
    memory = agent.memory.prepare(calculate_advantages=True)
    losses = {}

    # initialize test variables
    kl_target_reached = False

    # begin training loops
    n_epochs = agent.params.n_epochs
    for epoch in range(n_epochs):
        # Create the batches for training agent networks
        batches = agent.generate_batches()

        # 0. Fine-tune the penalty parameter
        if agent.has_penalty:
            agent.penalty.learn( memory.episode_costs )
            curr_penalty = agent.penalty.use_penalty()
            losses.update({'penalty': curr_penalty})

        # 1. Train the predictor (if the agent has one)
        if agent.has_predictor:
            predictor_loss = learn_predictor(agent, memory, batches)
            losses.update(predictor_loss)

        # 2. Train critic model(s)
        critic_losses = learn_critics(agent, memory, batches)
        losses.update(critic_losses)

        # 3. Train the Actor
        actor_loss, kl_target_reached, kl = learn_actor(agent, memory, batches)
        losses.update(actor_loss)

        if kl_target_reached:
            print(f"Early stopping at epoch {epoch} with KL divergence {kl}")
            break

    # post run, decay action std
    if agent.params.actions_continuous:
        agent.decay_action_std(0.01, 0.1)

    # post run, clear memory
    agent.memory.clear_memory()

    return losses

# 1. Train Predictor
#########################################################################################
def learn_predictor(agent, memory, batches):
    loss_fn = torch.nn.HuberLoss("mean") # torch.nn.MSELoss()
    predictor_losses = []

    for batch in batches:
        curr_states = memory.curr_states[batch]
        next_states = memory.next_states[batch]
        actions     = memory.actions[batch]

        # run predictor
        pred_states = agent.predictor(curr_states, actions)

        # Get loss
        predictor_loss = loss_fn(next_states, pred_states)

        # Update predictor
        [ model.zero_grad() for model in agent.models ]
        predictor_loss.backward()
        agent.predictor.optimizer.step()

        predictor_losses.append( predictor_loss.detach().cpu().numpy() )

    losses = {'predictor_loss': np.array(predictor_losses).mean()}
    return losses

# 2. Train Value Critic ( and Cost Critic )
#########################################################################################
def learn_critics(agent, memory, batches):
    loss_fn = torch.nn.MSELoss()
    #loss_fn = torch.nn.HuberLoss("mean")
    value_critic_losses = []
    cost_critic_losses  = []

    for batch in batches:
        curr_states  = memory.curr_states[batch]
        actions      = memory.actions[batch]
        returns      = memory.returns[batch]
        cost_returns = memory.cost_returns[batch]

        # 2.1 Run all the models to build gradients:
        # a) Run predictor, or just use current states
        # init variables used
        critic_loss = torch.tensor(0.)
        states      = curr_states

        if agent.has_predictor:
            with torch.no_grad():
                states = agent.predictor(curr_states, actions)

        # Get loss for value critic
        value_critic_value = agent.value_critic(states).squeeze()
        value_critic_loss  = loss_fn(returns, value_critic_value)
        value_critic_losses.append( value_critic_loss.detach().cpu().numpy() )
        critic_loss += value_critic_loss

        # Get loss for cost critic, if needed
        if agent.has_cost_critic:
            cost_critic_value = agent.cost_critic(states).squeeze()
            cost_critic_loss  = loss_fn(cost_returns, cost_critic_value)
            critic_loss += cost_critic_loss
            cost_critic_losses.append( cost_critic_loss.detach().cpu().numpy() )

        # 2.3 Backprop loss for critic(s)
        [ model.optimizer.zero_grad() for model in agent.models ]
        critic_loss.backward()
        agent.value_critic.optimizer.step()
        agent.cost_critic.optimizer.step() if agent.has_cost_critic else None

    losses = {}
    losses['value_critic_loss'] = np.array(value_critic_losses).mean()
    if agent.has_cost_critic:
        losses['cost_critic_loss']  = np.array(cost_critic_losses).mean()
    return losses

# 3. Train Actor
#########################################################################################
def learn_actor(agent, memory, batches):
    kl_target_reached = False
    actor_losses = []

    for batch in batches:
        curr_states     = memory.curr_states[batch]
        old_logprobs    = memory.logprobs[batch]
        actions         = memory.actions[batch]
        values          = memory.values[batch]
        cost_values     = memory.cost_values[batch]
        advantages      = memory.advantages[batch]
        cost_advantages = memory.cost_advantages[batch]

        # 3.1 Calculate difference in policy at the moment
        new_logprobs, entropies = agent.actor.calculate_entropy(curr_states, actions)

        # 3.2 Calculate KL divergence of Actor policy. ( Early Stop if Necessary )
        if not kl_target_reached:
            do_early_stop, kl = agent.check_kl_early_stop()
            if do_early_stop:
                kl_target_reached = True
                break

        # 3.4 If using predictors, calculate slightly modified advantages w/ gradient
        if agent.has_predictor:
            states                  = agent.predictor(curr_states, actions)
            value_critic_values     = agent.value_critic(states).squeeze()
            value_deltas  = values - value_critic_values
            value_deltas /= memory.advantages_scaling
            value_deltas  = value_deltas.clamp(-0.05, 0.05)
            advantages += value_deltas

        if agent.has_predictor and agent.has_cost_critic:
            cost_critic_values = agent.cost_critic(states).squeeze()
            cost_advantages += ( cost_values - cost_critic_values )

        # 3.5 calculate scaled advantages
        prob_ratio = ( new_logprobs - old_logprobs ).exp() # Likelihood ratio
        clip = agent.params.policy_clip
        surrogate_advantages = advantages * prob_ratio # scaled advantage

        if agent.params.clipped_advantage: # PPO
            scaled_clipped_advantages = \
                torch.clamp(prob_ratio, 1-clip, 1+clip) * advantages
            surrogate_advantages = \
                torch.min(surrogate_advantages, scaled_clipped_advantages)

        actor_objective = surrogate_advantages.mean()

        # Sometimes use cost advantages too. See safety-starter-agents
        if agent.has_cost_critic:
            curr_penalty = agent.penalty.use_penalty()
            surrogate_cost = (prob_ratio * cost_advantages).mean()
            actor_objective -= curr_penalty * surrogate_cost
            actor_objective /= (1 + curr_penalty)

        # Loss for actor policy is negative objective
        actor_loss = - actor_objective

        # Optionally, calculate entropy loss term
        entropy_loss = 0
        entropy_regularization = agent.params.entropy_regularization
        if entropy_regularization != 0:
            entropy_loss = entropy_regularization * entropies.mean()

        total_loss = actor_loss + entropy_loss

        # 3.6 Backprop loss for actor
        [ model.optimizer.zero_grad() for model in agent.models ]
        total_loss.backward()
        agent.actor.optimizer.step()

        actor_losses.append( total_loss.detach().cpu().numpy() )


    losses = { "actor_loss": np.array(actor_losses).mean() }
    return losses, kl_target_reached, kl
