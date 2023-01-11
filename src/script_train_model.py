""" Script for initializing parameters and training a new model
"""

from typing import Optional
from collections import defaultdict
import argparse
import json
import time
import csv
import os

# pylint: disable=import-error
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from memory import Memory, map_and_flatten_state
from trainer import trainer

from plot_scores import plot_scores

from agents import S3Agent, ActorCriticAgent
from world import CreateWorld
from params import Params

# pylint: disable=wildcard-import, unused-wildcard-import
from constraints import *

def main( game_mode: str,
          agent_type: str,
          model_name: str = "model",
          num_agents_to_train: int = 1,
          render: bool = False
        ):
    # Choose the environment
    if game_mode == "car":
        env = CreateWorld()
        num_timesteps = 1e7

        actions_continuous = True
        num_iter      = 30000
        batch_size    = 100
        num_epochs    = 80
        save_period   = num_iter*5
        run_tests     = False
        test_period   = np.Inf
        test_iter     = 0
        timestep_length = 1
        hidden_size   = 256
        reward_decay  = 0.99
        gae_lambda    = 0.97
        cost_decay    = 0.99
        cost_lambda   = 0.97
        learning_rate = 0.001
        kl_target     = 0.012

        calculate_constraint = calculate_constraint_cargoal2_v0

    elif game_mode == "cartpole":
        env = gym.make('CartPole-v1')
        num_timesteps = 100000

        actions_continuous = False
        num_iter      = 100
        batch_size    = 20
        num_epochs    = 4
        save_period   = num_iter*5
        run_tests     = True
        test_period   = 400
        test_iter     = 1000
        timestep_length = 1
        hidden_size   = 64
        reward_decay  = 0.99
        gae_lambda    = 0.95
        reward_decay  = 0.99
        gae_lambda    = 0.95
        learning_rate = 0.0003
        kl_target     = 0

        calculate_constraint = calculate_constraint_cartpole

    else:
        raise ValueError("game_mode must be 'car' or 'cartpole'")

    # pylint: disable=missing-class-docstring
    class S3AgentConstructor(S3Agent):
        def calculate_constraint( self, index: int,
                state: torch.Tensor, memory: Optional[Memory] = None ):
            return calculate_constraint( self, index, state, memory )

    # pylint: disable=missing-class-docstring
    class ActorCriticAgentConstructor(ActorCriticAgent):
        def calculate_constraint(self, index: int,
                state: torch.Tensor, memory: Optional[Memory] = None ):
            return calculate_constraint( self, index, state, memory )

    # Choose the Agent
    if agent_type == "s3":
        agent_constructor = S3AgentConstructor
    elif agent_type == "ac":
        agent_constructor = ActorCriticAgentConstructor
    else:
        raise ValueError("Invalid agent type")

    # initialise the world state
    curr_state, state_mapping = map_and_flatten_state( env.reset() )
    print( "# State mapping:")
    print( json.dumps(state_mapping, indent=4) )

    # get parameters needed to construct the agent
    state_size  = curr_state.size()[0]
    action_size = \
        env.action_space.n if not actions_continuous else env.action_space.shape[0]

    params = Params(
        state_size=state_size,
        action_size=action_size,
        hidden_size1=hidden_size,
        hidden_size2=hidden_size,
        actions_continuous=actions_continuous,
        use_cuda=torch.cuda.is_available(),
        learning_rate=learning_rate,
        reward_decay=reward_decay,
        gae_lambda=gae_lambda,
        clipped_advantage=True,
        policy_clip=0.2,
        action_std_init=0.4,
        kl_target=kl_target,
        cost_decay=cost_decay,
        cost_lambda=cost_lambda,
        train_cost_critic=True,
        checkpoint_dir=f"tmp/models/{game_mode}/{agent_type}/{model_name}",
        agent_type=agent_type,
        game_mode=game_mode,
        num_timesteps=num_timesteps,
        num_iter=num_iter,
        batch_size=batch_size,
        n_epochs=num_epochs,
        run_tests=run_tests,
        test_period=test_period,
        test_iter=test_iter,
        timestep_length=timestep_length,
        save_period=save_period,
        cumulative_limit=5,
    )
    print("# Parameters:")
    print(json.dumps(params._json(), indent=4))

    os.makedirs(params.checkpoint_dir, exist_ok=True)

    train_data_log = []
    test_data_log  = []
    for _ in range(num_agents_to_train):
        # run the training
        time_str = time.strftime("%Y.%m.%d.%H:%M:%S", time.localtime())
        run_name = f"{game_mode}-{agent_type}-{time_str}"

        agent = agent_constructor(params)
        train_data, test_data = trainer( env, agent, params, run_name, render )

        train_data_log.append( train_data )
        test_data_log.append( test_data )

        # save the scores from this run
        for data_collection, name in [(train_data, "training"), (test_data, "test")]:
            for metric, data_dict in data_collection.items():
                print( metric, ":", data_dict )
                folder_name = f"runs/{model_name}"
                os.makedirs(folder_name, exist_ok=True)
                with open(f"{folder_name}/{name}-{metric}-{run_name}.csv", "w") as f:
                    writer = csv.writer(f)
                    for label, data in data_dict.items():
                        writer.writerow([label] + data)

                # plot the scores
                _fig = plot_scores( data_dict, window_size=10, label=metric )
                folder_name = f"figs/{model_name}"
                os.makedirs(folder_name, exist_ok=True)
                file_name = f"{folder_name}/{name}-{metric}-{run_name}.png"
                plt.title( metric )
                plt.savefig( file_name, dpi=300 )

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_mode',  type=str, default='cartpole')
    parser.add_argument('--agent_type', type=str, default='ac')
    parser.add_argument('--model_name', type=str, default='model')
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('-n', type=int, default=1)

    args = parser.parse_args()
    main( args.game_mode, args.agent_type, model_name=args.model_name,
          num_agents_to_train=args.n, render=args.render )