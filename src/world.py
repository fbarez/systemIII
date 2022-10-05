import safety_gym
import gym
from safety_gym.envs.engine import Engine
import numpy as np
import torch

config = {
    'robot_base': 'xmls/car.xml',
    'task': 'goal',
    'observe_goal_lidar': True,
    'observe_box_lidar': False,
    'observe_hazards': True,
    'observe_vases': True,
    'constrain_hazards': False,
    'observation_flatten': False,
    'lidar_max_dist': 5, #how far it can see
    'lidar_num_bins': 10,
    'hazards_num': 3,
    'vases_num': 1
}

def CreateWorld():
    #env = Engine(config)
    env = gym.make('Safexp-CarGoal2-v0')

    # ensure state is returned as a dict
    state = env.reset()
    if not type(state) is dict:
        env.toggle_observation_space()
        state = env.reset()
    assert type(state) is dict

    return env