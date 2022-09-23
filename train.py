#from agents import *
#from envs import *
from utils import *
from config import *
from torch.multiprocessing import Pipe
from model import S3Model
from world import CreateWorld, flatten_state

from tensorboardX import SummaryWriter

import numpy as np
import copy
import torch
from dist_gen import get_distance

def main():
    """
    print({section: dict(config[section]) for section in config.sections()})
    train_method = default_config['TrainMethod']
    env_id = default_config['EnvID']
    env_type = default_config['EnvType']

    #if env_type == 'mario':
    #    env = BinarySpaceToDiscreteSpaceEnv(gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)
    #elif env_type == 'atari':
    #    env = gym.make(env_id)
    if env_type == 'safety':
        env = gym.make(env_id)
    else:
        raise NotImplementedError
    # Assert the observation is only one dimenstion
    assert len(env.observation_space.shape)==1 
    """
    env = CreateWorld()
    curr_state_dict = env.reset()

    curr_state, state_mapping = flatten_state(curr_state_dict, return_mapping=True)
    state_size = curr_state.size()[0]
    #input_size  = env.observation_space.shape[0]  #
    action_size = env.action_space.sample().size #.n  # 2
    print("state_size: ", state_size)
    print("action_size: ", action_size)

    env.close()
    runner = get_distance( env )

    use_cuda = default_config.getboolean('UseGPU')

    agent = S3Model
    
    hidden_size1=265
    hidden_size2=64
    agent = agent(
        state_size,
        action_size,
        hidden_size1,
        hidden_size2,
        use_cuda=use_cuda,
        state_mapping=state_mapping 
    )

    episode = 0
    while True:
        # Step 1. n-step rollout
        actions, curr_observations, next_observations, rewards, dones, curr_state_dict = runner.n_step_rollout( agent.choose_action, curr_state_dict, num_iter=100, render=True )
        """
        print("actions: ", actions)
        print("curr_observations: ", curr_observations)
        print("next_observations: ", next_observations)
        print("rewards: ", rewards)
        print("dones: ", dones)
        print("curr_state: ", curr_state)
"""
        #distances, weight_literal, probs, agent_positions, hazard_positions, actions = runner.extract_distances(next_observations, dones)

        # Step 5. Training!
        predictor_loss = agent.train_predictor( curr_observations, actions, next_observations )
        actor_loss = 0

        #plot results -- adobt
        print('episode', episode, 'predictor_loss', float(predictor_loss), 'actor_loss', float(actor_loss) )
        episode += 1

if __name__ == '__main__':
    main()
