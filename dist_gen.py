# #Author Fazl Barez

import safety_gym
import gym
import time
import numpy as np
import pandas as pd
from safety_gym.envs.engine import Engine
import warnings
#import util
#from util import *
print("ignore...\n")

warnings.filterwarnings('ignore')
config = {
    'robot_base': 'xmls/car.xml',
    'task': 'goal',
    'observe_goal_lidar': True,
    'observe_box_lidar': False,
    'observe_hazards': True,
    'observe_vases': False,
    'constrain_hazards': False,
    'observation_flatten': False,
    'lidar_max_dist': 5, #how far it can see
    'lidar_num_bins': 1,
    'hazards_num': 3,
    'vases_num': 1
}

# , 'Position Agent', next_observation['observe_qpos']
env = Engine(config)
def to_flatten_list(l):
    flat_list = []
    try:
        for element in l:
            try:
                len(element)
                flat_list.append(element)
            except:
                flat_list.extend(to_flatten_list(element))
    except:
        return [l]
            
    return flat_list

def random_action_sampler( env ):
    action = env.action_space.sample()
    return action    

class get_distance(object):
    def __init__(self, env):
        self.env = env
        return

    def dp_net(self, action_sampler=random_action_sampler, num_iter=100):
        """[summary]
        Args:
            state ([type]): [description]
            weights ([type]): [description]
            state_values ([type]): [description]
            probs ([type]): [description]
        Returns:
            [type]: [description]
        """        
        #env.reset()
        distances = []
        state_values = []
        current_observations = []
        current_observations_flat = []
        rewards = []
        agent_positions = []
        hazard_positions = []
        probs = []
        #state_values = []
        actions = []

        current_state = env.reset()
        
        for _ in range(num_iter):
            # initialise next state
            current_observations.append(current_state)            

            action = action_sampler( env )
            actions.append(action)
            output = next_observation, reward, done, info = env.step(action)
            #print(f"\n\noutput {_}:\n", [ o for o in output ])
            
            temp = []
            for item in current_state.values():
                temp.extend(item.flatten())
            current_observations_flat.append(temp)
            rewards.append(reward)
            distances.append((1-next_observation['hazards_lidar'])*config['lidar_max_dist'])
            
            '''
            Return a robot-centric lidar observation of a list of positions.
            Lidar is a set of bins around the robot (divided evenly in a circle).
            The detection directions are exclusive and exhaustive for a full 360 view.
            Each bin reads 0 if there are no objects in that direction.
            If there are multiple objects, the distance to the closest one is used.
            Otherwise the bin reads the fraction of the distance towards the robot.
            E.g. if the object is 90% of lidar_max_dist away, the bin will read 0.1,
            and if the object is 10% of lidar_max_dist away, the bin will read 0.9.
            (The reading can be thought of as "closeness" or inverse distance)
            This encoding has some desirable properties:
            - bins read 0 when empty
            - bins smoothly increase as objects get close
            - maximum reading is 1.0 (where the object overlaps the robot)
            - close objects occlude far objects
            - constant size observation with variable numbers of objects
            '''
            probs.append(next_observation['hazards_lidar'])
            # env.render() #uncomment if you want to visulaize 

            
            all_observations = {} 
            # get the state describtion
            for key in next_observation.keys(): 
                if len(next_observation[key].shape)>1:continue
                all_observations[key] = next_observation[key] 
            all_observations = np.concatenate([all_observations[key] for key in all_observations.keys()]) 
            state_values.append(all_observations)
                
            #converting dict into list 
            for element in next_observation.items():
                for item in element[1]:
                    state_values.extend(to_flatten_list(item))
            
            #do the same operation for the current state 
            for element in current_state.values():
                current_observations_flat.append(item.flatten())

            #turn the state description dictionary into a list 
            for key, value in next_observation.items():
                state_values = [key, value]
                state_values.append(state_values)
                
            #get agent's current position
            agent_positions.append(env.world.robot_pos())
            
            #get position of the hazards
            for h_pos in env.hazards_pos:
                hazard_positions.append(h_pos)

            # get things ready for next loop
            if done:
                current_state = env.reset()
            else:
                current_state = next_observation

        distances      = np.array(distances)
        state_values   = np.array(state_values)
        weight_literal = 1 - distances
        probs = np.array(probs)
        hazard_positions = np.array(hazard_positions)
        agent_positions  = np.array(agent_positions)
        actions = np.array(actions)
        current_observations_flat = np.array(current_observations_flat)
        
        #print("State Values:", state_values)
        #print("reward we want:", np.array([ rewards[i]*distances[i] for i in range(len(rewards)) ]).flatten() )
        return distances, weight_literal, state_values, current_observations_flat, probs, actions, agent_positions, hazard_positions, 

# print("Running get_distance():")
distances, weights, state_value, current_state, probs, agent_positions, hazard_positions, actions = get_distance(env).dp_net(random_action_sampler, 1000)
# print("Running get_distance() completed successfully")
# #print("Distances to objects:", distances, "Weight of Literals:", weights, "State observation values:", state_value, "Agent Position:", agent_position)