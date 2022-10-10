from tkinter import W
from config import *
from torch.multiprocessing import Pipe
from agents import S3Agent, ActorCriticAgent, Agent
from world import CreateWorld
from params import Params

from typing import Optional

import numpy as np
import copy
import torch
import time
from dist_gen import get_distance
from memory import Memory
import gym
import csv
import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from plot_scores import plot_scores

def train_model( env, agent:Agent, params:Params, run_name:str ):
    runner = get_distance( env, agent )
    curr_state = Memory().flatten_state( env.reset() )

    episode, timesteps = 0, 0
    render    = False
    default_scores_dict = lambda : {"t": [], "episode":[], "val": [], "mean": [], "std": []}
    train_run_data = defaultdict(default_scores_dict)
    test_run_data  = defaultdict(default_scores_dict)
    train_states = []
    test_states  = []
    run_data = None
    t0 = time.time()

    # generate titles for states and constraints csv
    curr_state, state_mapping = Memory().flatten_state( env.reset(), return_mapping=True )
    reverse_mapping = ["" for _ in range(params.state_size)]
    for key, [start,end] in state_mapping.items():
        for i in range(start, end):
            reverse_mapping[i] = key+"_"+str(i-start)
    action_mapping = ["action" if not params.actions_continuous else "action_"+str(i) for i in range(params.action_size)]
    titles = ["time", "done", "reward", "constraint", *action_mapping, *reverse_mapping ]
    train_states.append( titles )
    test_states.append( titles )
    for state_data, name in [(train_states, "training"), (test_states, "test")]:
        if name == "test" and not params.run_tests:
            continue
        # save the states from this run
        with open(f"runs/{name}-states-{run_name}.csv", "a") as f:
            writer = csv.writer(f)
            for state in state_data:
                writer.writerow(state)
    train_states, test_states = [], []

    # pre-test agent as a zero datapoint
    # TODO: Add code for testing agent again

    while timesteps < params.num_timesteps:
        # Step 1. n-step rollout
        curr_state, run_data, state_data = runner.n_step_rollout( agent.choose_action,
            curr_state, params.num_iter, render, prev_run_data=run_data, training=True )
        train_states += [ [timesteps+i, *state] for i, state in enumerate(state_data) ]
        timesteps += params.num_iter

        # step 2. get some info
        num_dones = np.sum( agent.memory.dones )

        # Step 3. Learn
        losses = agent.learn()
        losses = { k:round(float(v),2) for k,v in losses.items()}        

        if agent.params.actions_continuous:
            agent.decay_action_std(0.01, 0.1)

        # Step 4. Print some info about the training so far
        initial_episode = episode
        num_dones = len( run_data["score"] ) - 1
        episode += num_dones

        for metric, data in run_data.items():
            for i, value in enumerate(data):
                train_run_data[metric]["val"].append( value )
                train_run_data[metric]["episode"].append( initial_episode + i )

        scores = run_data["score"]
        for i, score in enumerate(scores[:-1]):
            loss_info = f'\tlosses {losses}' if i==num_dones-1 else ''
            print('episode', initial_episode+i, 'score %.2f' % score, loss_info )
        
        # Step 5. Test the model to gain insight into performance
        # TODO: Re-add code for testing the model

        # Step 6. Save the model
        if params.save_period and timesteps % params.save_period == 0:
            agent.save_models()

        # Step 7. Save logs about the Model states
        for state_data, name in [(train_states, "training"), (test_states, "test")]:
            if name == "test" and not params.run_tests:
                continue
            # save the states from this run
            with open(f"runs/{name}-states-{run_name}.csv", "a") as f:
                writer = csv.writer(f)
                for state in state_data:
                    writer.writerow(state)
        
        train_states, test_states  = [], []
        
    print("Finished training")
    return train_run_data, test_run_data

def main( game_mode:str, agent_type:str, model_name:str="model", num_agents_to_train:int=1 ):
    # Choose the environment
    if game_mode == "car":
        env = CreateWorld()
        num_timesteps = 1e5

        actions_continuous = True
        num_iter      = 3000
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
        learning_rate = 0.001
        kl_target     = 0.012

        def calculate_constraint( self:Agent, state:torch.Tensor, memory:Optional[Memory]=None ):
            max_lidar_range = 5
            memory = memory if not memory is None else self.memory
            hazards_lidar = memory.flat_get( state, 'hazards_lidar' )
            closest_hazard = ( 1 - torch.max(hazards_lidar) )*max_lidar_range
            hazardous_distance = 0.5
            constraint = ( 1 - torch.clamp( closest_hazard, 0, hazardous_distance )*(1/hazardous_distance) )
            return 1 - constraint

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
        gae_lambda    = 0.95
        learning_rate = 0.0003
        kl_target     = 0
        
        def calculate_constraint( self:Agent, state:torch.Tensor, memory:Optional[Memory]=None ):
            pole_angle, pole_max = state[2], 0.2095
            hazardous_distance = 0.1
            angle_remaining = pole_max - torch.abs(pole_angle)
            constraint = ( 1 - torch.clamp(angle_remaining, 0, hazardous_distance)*(1/hazardous_distance) )
            return 1 - constraint

    else:
        raise ValueError("game_mode must be 'car' or 'cartpole'")
    
    class S3AgentConstructor(S3Agent):
        def calculate_constraint( self, state:torch.Tensor, memory:Optional[Memory]=None ):
            return calculate_constraint( self, state, memory )
    class ActorCriticAgentConstructor(ActorCriticAgent):
        def calculate_constraint( self, state:torch.Tensor, memory:Optional[Memory]=None ):
            return calculate_constraint( self, state, memory )

    # Choose the Agent
    if agent_type == "s3":
        agent_constructor = S3AgentConstructor
    elif agent_type == "ac":
        agent_constructor = ActorCriticAgentConstructor
    else:
        raise ValueError("Invalid agent type")

    # initialise the world state
    curr_state, state_mapping = Memory().flatten_state( env.reset(), return_mapping=True )
    print( state_mapping, "\n" )

    # get parameters needed to construct the agent
    state_size  = curr_state.size()[0]
    action_size = env.action_space.n if not actions_continuous else env.action_space.shape[0]
    print(" state_size:  ", state_size, "\n action_size: ", action_size)

    params = Params(
        state_size=state_size,
        action_size=action_size,
        hidden_size1=hidden_size,
        hidden_size2=hidden_size,
        actions_continuous=actions_continuous,
        learning_rate=learning_rate,
        reward_decay=0.99,
        gae_lambda=gae_lambda,
        policy_clip=0.2,
        action_std_init=0.4,
        kl_target=kl_target,
        use_cuda=torch.cuda.is_available(),
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
    )
    os.makedirs(params.checkpoint_dir, exist_ok=True)

    train_data_log = []
    test_data_log  = []
    for i in range(num_agents_to_train):
        # run the training
        time_str = time.strftime("%Y.%m.%d.%H:%M:%S", time.localtime())
        run_name = f"{game_mode}-{agent_type}-{time_str}"

        agent = agent_constructor(params)
        train_data, test_data = train_model( env, agent, params, run_name ) 

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
                fig = plot_scores( data_dict, window_size=10, label=metric )
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
    parser.add_argument('-n', type=int, default=1)

    args = parser.parse_args()
    main( args.game_mode, args.agent_type, model_name=args.model_name, num_agents_to_train=args.n )
