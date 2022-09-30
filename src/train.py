#from agents import *
#from envs import *
from tkinter import W
from utils import *
from config import *
from torch.multiprocessing import Pipe
from agents import S3Agent, ActorCriticAgent, Agent
from world import CreateWorld
from params import Params

from tensorboardX import SummaryWriter
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

import matplotlib.pyplot as plt
from plot_scores import plot_scores

def train_model( env, agent:Agent, params:Params ):
    runner = get_distance( env, agent )
    curr_state = Memory().flatten_state( env.reset() )

    episode, timesteps = 0, 0
    render    = False
    scores    = [0]
    train_scores = {"t": [], "episode":[], "val": [], "mean": [], "std": []}
    test_scores  = {"t": [], "episode":[], "val": [], "mean": [], "std": []}
    train_states = []
    test_states  = []
    t0 = time.time()

    # generate titles for constraints csv
    curr_state, state_mapping = Memory().flatten_state( env.reset(), return_mapping=True )
    reverse_mapping = ["" for _ in range(params.state_size)]
    for key, [start,end] in state_mapping.items():
        for i in range(start, end):
            reverse_mapping[i] = key+"_"+str(i-start)
    action_mapping = ["action" if not params.actions_continuous else "action_"+str(i) for i in range(params.action_size)]
    titles = ["time", "done", "reward", "constraint", *action_mapping, *reverse_mapping ]
    train_states.append( titles )
    test_states.append( titles )

    # pre-test agent as a zero datapoint
    test_index  = 0
    save_models = True
    
    print_test_scores = lambda s : print( " -- test scores:", round(s["mean"][-1], 2), "pm", round(s["std"][-1], 2) )

    # test agent
    curr_state, scores, state_data = runner.n_step_rollout( agent.choose_action,
        curr_state, params.test_iter, render, training=False )
    test_scores[ "mean" ].append( np.mean(scores[:-1]) )
    test_scores[ "std"  ].append( np.std(scores[:-1]) )
    test_scores[ "t"    ].append( timesteps*params.timestep_length )
    print_test_scores( test_scores )
    if save_models:
        agent.save_models()
    test_states += [ [test_index, *state] for state in state_data ]
    curr_state  = Memory().flatten_state( env.reset() )

    scores[-1] = 0
    test_index += 1
    

    while timesteps < params.num_timesteps:
        # Step 1. n-step rollout
        curr_state, scores, state_data = runner.n_step_rollout( agent.choose_action,
            curr_state, params.num_iter, render, prev_score=scores[-1], training=True )
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
        for i, score in enumerate(scores[:-1]):
            episode += 1
            train_scores[ "val"     ].append( round(score, 3) )
            train_scores[ "episode" ].append( episode )
            loss_info = f'\tlosses {losses}' if i==num_dones-1 else ''
            print('episode', episode, 'score %.2f' % scores[i], loss_info )
        
        # Step 5. Test the model to gain insight into performance
        if timesteps % params.test_period == 0:
            # test agent
            curr_state = Memory().flatten_state( env.reset() )
            curr_state, scores, state_data = runner.n_step_rollout( agent.choose_action,
                curr_state, params.test_iter, render, training=False )
            test_scores[ "mean" ].append( np.mean(scores[:-1]) )
            test_scores[ "std"  ].append( np.std(scores[:-1]) )
            test_scores[ "t"    ].append( timesteps*params.timestep_length )
            print_test_scores( test_scores )
            if save_models:
                agent.save_models()
            test_states += [ [test_index, *state] for state in state_data ]

            curr_state = Memory().flatten_state( env.reset() )
            scores[-1] = 0
            test_index += 1
        
        #print( "train states:\n", [ state[3] for state in train_states ])
        #print( "\ntest states:\n", [ state[3] for state in test_states ] )

    print("Finished training")
    return train_scores, test_scores, train_states, test_states

def main( game_mode:str, agent_type:str, num_agents_to_train:int=1 ):
    # Choose the environment
    if game_mode == "car":
        env = CreateWorld()
        num_timesteps = 128*1024

        actions_continuous = True
        num_iter    = 2048
        batch_size  = 64
        num_epochs  = 10
        test_period = 2048*2
        test_iter   = 2048
        timestep_length = 10

        class S3AgentConstructor(S3Agent):
            def calculate_constraint( self, state:torch.Tensor, memory:Optional[Memory]=None ):
                max_lidar_range = 5
                memory = memory if not memory is None else self.memory
                hazards_lidar = memory.flat_get( state, 'hazards_lidar' )
                closest_hazard = ( 1 - torch.max(hazards_lidar) )*max_lidar_range
                hazardous_distance = 0.5
                constraint = ( 1 - torch.clamp( closest_hazard, 0, hazardous_distance )*(1/hazardous_distance) )
                return 1 - constraint

        class ActorCriticAgentConstructor(ActorCriticAgent):
            def calculate_constraint( self, state:torch.Tensor, memory:Optional[Memory]=None ):
                max_lidar_range = 5
                memory = memory if not memory is None else self.memory
                hazards_lidar = memory.flat_get( state, 'hazards_lidar' )
                closest_hazard = ( 1 - torch.max(hazards_lidar) )*max_lidar_range
                hazardous_distance = 0.5
                constraint = ( 1 - torch.clamp( closest_hazard, 0, hazardous_distance )*(1/hazardous_distance) )
                return 1 - constraint

    elif game_mode == "cartpole":
        env = gym.make('CartPole-v0')
        num_timesteps = 20000

        actions_continuous = False
        num_iter    = 20
        batch_size  = 5
        num_epochs  = 4
        test_period = 25*20
        test_iter   = 1000
        timestep_length = 1
        
    else:
        raise ValueError("game_mode must be 'car' or 'cartpole'")

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
        hidden_size1=256,
        hidden_size2=256,
        actions_continuous=actions_continuous,
        learning_rate=0.0003,
        reward_decay=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        action_std_init=0.4,
        use_cuda=torch.cuda.is_available(),
        agent_type=agent_type,
        game_mode=game_mode,
        num_timesteps=num_timesteps,
        num_iter=num_iter,
        batch_size=batch_size,
        n_epochs=num_epochs,
        test_period=test_period,
        test_iter=test_iter,
        timestep_length=timestep_length,
    )

    train_scores_log = []
    test_scores_log  = []
    train_states_log = []
    test_states_log  = []
    for _ in range(num_agents_to_train):
        # run the training
        agent = agent_constructor(params)
        train_scores, test_scores, train_states, test_states = train_model( env, agent, params ) 

        # generate unique string for this run
        time_str = time.strftime("%Y.%m.%d.%H:%M:%S", time.localtime())
        run_name = f"{game_mode}-{agent_type}-{time_str}"

        train_scores_log.append( train_scores )
        test_scores_log.append( test_scores )
        train_states_log.append( train_states )
        test_states_log.append( test_states )

        for score_data, name in [(train_scores, "training"), (test_scores, "test")]:
            # save the scores from this run
            print( score_data )
            with open(f"runs/{name}-scores-{run_name}.csv", "w") as f:
                writer = csv.writer(f)
                for label, data in score_data.items():
                    writer.writerow([label] + data)

            # plot the scores
            fig = plot_scores( score_data, window_size=10 )
            plt.savefig( f"figs/{name}-{run_name}.png", dpi=300 )

        for state_data, name in [(train_states, "training"), (test_states, "test")]:
            # save the states from this run
            with open(f"runs/{name}-states-{run_name}.csv", "w") as f:
                writer = csv.writer(f)
                for state in state_data:
                    writer.writerow(state)

    test_scores_dict = {
        "mean": np.mean( [ d["mean"] for d in test_scores_log ], axis=0 ),
        "std":  np.std(  [ d["mean"] for d in test_scores_log ], axis=0 ),
        "t":    np.mean( [ d["t"]    for d in test_scores_log ], axis=0 ),
        "val":  [],
        "episode": [],
    }
    # plot the scores
    print( test_scores_dict )
    fig = plot_scores( test_scores_dict, window_size=10 )
    plt.savefig( f"figs/{name}-{run_name}.png", dpi=300 )

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_mode',  type=str, default='cartpole')
    parser.add_argument('--agent_type', type=str, default='ac')
    parser.add_argument('-n', type=int, default=1)

    args = parser.parse_args()
    main( args.game_mode, args.agent_type, num_agents_to_train=args.n )
