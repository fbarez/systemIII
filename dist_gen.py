#Author Fazl Barez

import safety_gym
import gym
import time
import numpy as np
import pandas as pd
from safety_gym.envs.engine import Engine
import warnings
#import util
#from util import *


warnings.filterwarnings('ignore')
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
    'lidar_num_bins': 1,
    'hazards_num': 0,
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
    
class get_distance(object):
    def __init__(self):
        
        #self.obj_type = obj_type
        return  
    def dp_net(self, state, weights, state_values, probs):
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
        current_observation = []
        current_observation_flat = []
        agent_position = []
        hazard_position = []
        probs = []
        #state_values = []
        actions = []

        for _ in range(100):
            state = env.reset()
            #print("state:", state) 
            action = env.action_space.sample()
            current_observation.append(state)
            actions.append(action)
            next_observation, reward, done, info = env.step(action)
            temp = []
            for item in state.values():
                temp.extend(item.flatten())
            current_observation_flat.append(temp)
            distances.append((1-next_observation['hazards_lidar'])*config['lidar_max_dist'])
            if done:
                env.reset()
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
            #env.render() #uncomment if you want to visulaize 


        #     all_observations = {} 
        #    # get the state describtion
        #     for key in next_observation.keys(): 
        #         if len(next_observation[key].shape)>1:continue
        #         #print("KEY:", key)     
        #         all_observations[key] = next_observation[key] 
        #     all_observatsions = np.concatenate([all_observations[key] for key in all_observations.keys()]) 
        #     state_values.append(all_observations)
                
            #converting dict into list 
            for element in next_observation.items():
                for item in element[1]:
                    # import pdb
                    # pdb.set_trace()
                    state_values.extend(to_flatten_list(item))
            
            #print(type(current_observation))
            # for element in current_observation:
            #     print(element)
            
            #do the sameoperation for the current state 
            # for element in current_observation:
            #     #print("ELEMENTS:", element)
            #     for item in element.values():
            #         current_observation_flat.append(item.flatten())
                    #import pdb; pdb.set_trace()
                #print("Items is:", i[1])
            #turn the state description dictionary into a list 
            # for key, value in next_observation.keys():
            #     state_values = [key, value]
            #     state_values.append(state_values)
                
            #get agent's current position
            agent_position.append(env.world.robot_pos())
            
            #get position of the hazards
            for h_pos in env.hazards_pos:
                hazard_position.append(h_pos)
            #hazard_position.append(env.world.hazards_pos)
        #print("Lets debug this"))
            #for the case Float
            #all_observations = np.concatenate([next_observation for )
            #state_values.append(all_observations)
        #distance = next_observation['hazards_lidar']
        distances = np.array(distances)
        state_values = np.array(state_values)
        weight_literal = 1 - distances
        probs = np.array(probs)
        hazard_position = np.array(hazard_position)
        #agent_position = env.world.robot_pos()
        agent_position = np.array(agent_position)
        actions = np.array(actions)
        current_observation_flat = np.array(current_observation_flat)
        #print("Lets debug this")
        #import pdb; pdb.set_trace()
        #print("State Values:", state_values)
        return distances, weight_literal, state_values, current_observation_flat, probs, actions, agent_position, hazard_position, 
distances, weights, state_value, current_state, probs, agent_position, hazard_position, actions = get_distance().dp_net(1, 1, 1, 1)
#print("Distances to objects:", distances, "Weight of Literals:", weights, "State observation values:", state_value, "Agent Position:", agent_position)