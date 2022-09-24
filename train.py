#from agents import *
#from envs import *
from utils import *
from config import *
from torch.multiprocessing import Pipe
from agents import S3Agent, ActorCriticAgent
from world import CreateWorld

from tensorboardX import SummaryWriter

import numpy as np
import copy
import torch
from dist_gen import get_distance
from memory import Memory

def main():
    env = CreateWorld()
    memory = Memory()

    # initialise the world state
    curr_state, state_mapping = memory.flatten_state( env.reset(), return_mapping=True )

    state_size  = curr_state.size()[0] 
    action_size = env.action_space.sample().size 
    print("state_size:  ", state_size)
    print("action_size: ", action_size)

    use_cuda = default_config.getboolean('UseGPU')

    agent = ActorCriticAgent
    
    hidden_size1=265
    hidden_size2=64
    agent = agent(
        state_size,
        action_size,
        hidden_size1,
        hidden_size2,
        use_cuda=use_cuda,
        memory=memory,
    )

    runner = get_distance( env, agent )

    episode = 0
    while True:
        # Step 1. n-step rollout
        curr_state = runner.n_step_rollout( agent.choose_action, curr_state, num_iter=128, render=True )

        # Step 2. Learn
        actor_loss, predictor_loss = agent.learn()

        #plot results -- adobt
        print('episode', episode, 'predictor_loss', float(predictor_loss),
                                  'actor_loss', float(actor_loss) )
        episode += 1

if __name__ == '__main__':
    main()
