""" Define learning functions for training the model
"""

import torch
from tqdm import tqdm
from agents import Agent

def learn(agent: Agent):
    # prepare advantages and other tensors used for training
    memory = agent.memory.prepare(calculate_advantages=True)

    # initialize test variables
    has_predictor = hasattr(agent, 'predictor')
    has_cost_critic = hasattr(agent, 'cost_critic')
    kl_target_reached = False

    # begin training loops
    n_epochs = agent.params.n_epochs
    for epoch in tqdm(range(n_epochs)):

        # Fine-tune the penalty parameter
        agent.penalty.learn( memory.episode_costs )
        curr_penalty = agent.penalty.use_penalty()

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
                entropy_loss = entropy_regularization * entropies.mean()

            # 2.7 Calculate total loss for actor and critics
            total_loss = actor_loss + 0.5*critic_loss + entropy_loss
            for model in agent.models:
                model.optimizer.zero_grad()

            # 2.8 Backprop loss for actor and critic(s)
            total_loss.backward()
            agent.actor.optimizer.step()
            agent.value_critic.optimizer.step()
            agent.cost_critic.optimizer.step() if hasattr(agent, 'cost_critic') else None

    # post run, decay action std
    if agent.params.actions_continuous:
        agent.decay_action_std(0.01, 0.1)

    # post run, clear memory
    agent.memory.clear_memory()

    losses = {}
    losses['actor'] = -actor_objective
    if has_predictor:
        losses['predictor'] = predictor_loss
    losses['value_critic'] = value_critic_loss
    if has_cost_critic:
        losses['cost_critic'] = cost_critic_loss

    return losses