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
from time import time
from dist_gen import get_distance
from memory import Memory
import gym

def main():
    env = CreateWorld()
    #env = gym.make('CartPole-v1')

    num_iter = 2048
    batch_size = 64

    # initialise the world state
    curr_state = env.reset()

    curr_state, state_mapping = Memory().flatten_state( env.reset(), return_mapping=True )
    print( state_mapping )

    state_size  = curr_state.size()[0] 
    #action_size = env.action_space.n
    action_size = env.action_space.sample().size 
    print("state_size:  ", state_size)
    print("action_size: ", action_size)

    use_cuda = default_config.getboolean('UseGPU')

    agent = S3Agent
    agent = ActorCriticAgent

    params = Params(
        state_size=state_size,
        action_size=action_size,
        hidden_size1=256,
        hidden_size2=256,
        actions_continuous=True,
        learning_rate=0.0003,
        reward_decay=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        action_std_init=0.4,
        batch_size=batch_size,
        n_epochs=10,
        use_cuda=use_cuda
    )

    agent = agent(params)

    runner = get_distance( env, agent )

    episode = 0
    render = False
    score = 0
    t0 = time()

    while True:
        # Step 1. n-step rollout
        curr_state = runner.n_step_rollout( agent.choose_action, curr_state, num_iter=num_iter, render=render )

        # step 2. get some info
        num_dones = np.sum( agent.memory.dones )

        # Step 3. Learn
        losses = agent.learn()
        losses = { k:round(float(v),2) for k,v in losses.items()}        

        if agent.params.actions_continuous:
            agent.decay_action_std(0.01, 0.1)

        if num_dones:
            print('episode', episode, '\tlosses', losses )
            episode += num_dones
            if time() - t0 > 2:
                t0 = time()
                render = episode
            elif render != episode:
                render = False 

if __name__ == '__main__':
    main()
