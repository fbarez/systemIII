""" Define the default parameters for the Safety Gym environment
"""
# pylint: disable=import-error
import gym
import safety_gym

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
    if not isinstance(state, dict):
        env.toggle_observation_space()
        state = env.reset()
    assert isinstance(state, dict)

    return env
