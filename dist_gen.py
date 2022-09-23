# #Author Fazl Barez
import warnings
warnings.filterwarnings('ignore')

import safety_gym
import gym
import time
import numpy as np
import pandas as pd
import torch
from world import CreateWorld, flatten_state
print("ignore...\n")

# , 'Position Agent', next_observation['observe_qpos']
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
    def __init__(self, env):
        self.env = env
        return

    def random_action_sampler( self, state ):
        action = env.action_space.sample()
        return action    

    def n_step_rollout( self, action_sampler=None, curr_state=None, num_iter=100, render=False ):
        """[summary]
        action_sampler: function that takes in a state and returns an action
        current_state: the initial state of the rollout
        num_iter: number of iterations to run the rollout
        render: whether to render the environment
        """
        # if no agent provided, use a random action sampler
        if action_sampler is None:
            action_sampler = self.random_action_sampler
        
        actions = []
        curr_observations_flat = []
        next_observations_flat = []
        rewards = []
        dones = []
        
        # initialise state
        if curr_state is None:
            curr_state = env.reset()

        for _ in range(num_iter):
            # get next state
            curr_state_flat = flatten_state(curr_state)
            action = action_sampler( curr_state_flat ).detach()
            output = next_observation, reward, done, info = self.env.step(action)
            #print(f"\n\noutput {_}:\n", [ o for o in output ])
            
            if render:
                self.env.render()

            # save information from episode
            actions.append(action)
            curr_observations_flat.append( curr_state_flat )
            next_observations_flat.append( flatten_state(next_observation) )
            rewards.append(reward)
            dones.append(done)

            # get things ready for next loop
            if done:
                current_state = self.env.reset()
            else:
                current_state = next_observation

        actions = torch.stack(actions)
        next_observations_flat = torch.stack(next_observations_flat)
        curr_observations_flat = torch.stack(curr_observations_flat)
        rewards = torch.tensor(rewards)
        dones   = torch.tensor(dones)

        return actions, curr_observations_flat, next_observations_flat, rewards, dones, curr_state

    def extract_distances(self, observations):
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
 
        #env.reset()
        distances = []
        agent_positions = []
        hazard_positions = []
        probs = []
        actions = []

        for state in observations: 
            distances.append((1-state['hazards_lidar'])*self.env.config['lidar_max_dist'])

            probs.append(state['hazards_lidar'])
            
            #get agent's current position
            agent_positions.append(self.env.world.robot_pos())
            
            #get position of the hazards
            for h_pos in self.env.hazards_pos:
                hazard_positions.append(h_pos)

        distances      = np.array(distances)
        weight_literal = 1 - distances
        probs = np.array(probs)
        hazard_positions = np.array(hazard_positions)
        agent_positions  = np.array(agent_positions)
        actions = np.array(actions)

        return distances, weight_literal, probs, agent_positions, hazard_positions, actions

if __name__ == "__main__":    
    print("\nRunning get_distance():")
    env = CreateWorld()
    runner = get_distance(env)
    actions, curr_observations, next_observations, rewards, dones, current_state = runner.n_step_rollout( num_iter=1000, render=True)
    distances, weight_literal, probs, agent_positions, hazard_positions, actions = runner.extract_distances(next_observations)
    print("State Values:", next_observations)
    print("State Values:", np.array([ flatten_state(x) for x in next_observations ]) )
    print("reward we want:", np.array([ rewards[i]*distances[i] for i in range(len(rewards)) ]).flatten()[:10] )
    # print("Running get_distance() completed successfully")
    # #print("Distances to objects:", distances, "Weight of Literals:", weights, "State observation values:", state_value, "Agent Position:", agent_position)