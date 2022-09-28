#from agents import *
#from envs import *
from utils import *
from config import *
from torch.multiprocessing import Pipe
from agents import S3Agent, ActorCriticAgent
from world import CreateWorld
from params import Params

from tensorboardX import SummaryWriter

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
from plot import plot_scores

def main( game_mode:str, agent_type:str ):

    if game_mode == "car":
        env = CreateWorld()
        num_episodes = 300
        num_iter = 2048
        batch_size = 64
        num_epochs = 10
        actions_continuous = True

    elif game_mode == "cartpole":
        env = gym.make('CartPole-v0')
        num_episodes = 300
        num_iter = 20
        batch_size = 5
        num_epochs = 4
        actions_continuous = False

    else:
        raise ValueError("game_mode must be 'car' or 'cartpole'")

    # initialise the world state
    curr_state = env.reset()

    curr_state, state_mapping = Memory().flatten_state( env.reset(), return_mapping=True )
    print( state_mapping )

    state_size  = curr_state.size()[0]
    if actions_continuous: 
        action_size = env.action_space.sample().size 
    else:
        action_size = env.action_space.n
    print("state_size:  ", state_size)
    print("action_size: ", action_size)

    use_cuda = default_config.getboolean('UseGPU')

    if agent_type == "s3":
        agent = S3Agent
    elif agent_type == "ac":
        agent = ActorCriticAgent
    else:
        raise ValueError("Invalid agent type")

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
        batch_size=batch_size,
        n_epochs=num_epochs,
        use_cuda=use_cuda
    )

    agent = agent(params)

    runner = get_distance( env, agent )

    episode = 0
    render = False
    scores = [0]
    all_scores = []
    t0 = time.time()

    while episode < num_episodes:
        # Step 1. n-step rollout
        curr_state, scores = runner.n_step_rollout( agent.choose_action,
            curr_state, num_iter, render, prev_score=scores[-1] )

        # step 2. get some info
        num_dones = np.sum( agent.memory.dones )

        # Step 3. Learn
        losses = agent.learn()
        losses = { k:round(float(v),2) for k,v in losses.items()}        

        if agent.params.actions_continuous:
            agent.decay_action_std(0.01, 0.1)

        for i, score in enumerate(scores[:-1]):
            episode += 1
            all_scores.append(round(score, 3))
            if i == num_dones-1:
                print('episode', episode, 'score %.2f' % score, '\tlosses', losses )
                
                if time.time() - t0 > 10:
                    t0 = time.time()
                    render = episode
                elif render != episode:
                    render = False 
            else:
                print('episode', episode, 'score %.2f' % scores[i] )

        render = False

    print("Finished training")

    # generate unique string for this run
    time_str = time.strftime("%Y.%m.%d.%H:%M:%S", time.localtime())
    run_name = f"scores-{game_mode}-{agent_type}-{time_str}"

    # save the scores from this run
    print( all_scores )
    with open(f"runs/{run_name}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(all_scores)

    # plot the scores
    fig = plot_scores( all_scores, window_size=10 )
    plt.savefig( f"figs/{run_name}.png", dpi=300 )
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_mode', type=str, default='cartpole')
    parser.add_argument('--agent_type', type=str, default='ac')

    args = parser.parse_args()
    main( args.game_mode, args.agent_type )
