# #Author Fazl Barez
from re import I
import warnings
warnings.filterwarnings('ignore')

import safety_gym
import gym
from time import time
import numpy as np
import pandas as pd
import torch
from world import CreateWorld
from memory import Memory
from typing import Optional, Callable
print("ignore...\n")

# , 'Position Agent', next_observation['observe_qpos']
class get_distance(object):
    def __init__(self, env, agent=None):
        self.env = env
        self.agent = agent
        self.memory = Memory() if agent is None else self.agent.memory
        return

    def random_action_sampler( self, state ):
        action = self.env.action_space.sample()
        return action, 0

    def n_step_rollout( self,
                        action_sampler: Optional[Callable] = None,
                        curr_state: Optional[torch.Tensor] = None,
                        num_iter: int = 100,
                        render: bool  = False,
                        prev_score: int = 0,
                        training: bool = False ):
        """[summary]
        action_sampler: function that takes in a state and returns an action
        current_state: the initial state of the rollout
        num_iter: number of iterations to run the rollout
        render: whether to render the environment
        """
        zero = lambda : torch.tensor(0)
        agent_has = lambda attr : hasattr(self.agent, attr)
        scores = [ prev_score ]

        # if no agent provided, use a random action sampler
        if action_sampler is None:
            action_sampler = self.random_action_sampler

        # initialise state
        if curr_state is None:
            curr_state = self.memory.flatten_state( self.env.reset() )
        
        with torch.no_grad():
            
            for _ in range(num_iter):
                # Get the next state
                action, action_logprob = action_sampler( curr_state, training )
                [ next_state, reward, done, info ] = self.env.step(np.array( action ))
                next_state = self.memory.flatten_state(next_state)
                value = self.agent.critic(curr_state) if agent_has('critic') else zero()
                pred_state = self.agent.predictor(curr_state, action) if agent_has('predictor') else zero()

                # Store the transition
                self.memory.add(curr_state, next_state, pred_state, 
                                action, action_logprob, reward, value, done)
                scores[-1] += reward

                if render:
                    self.env.render()

                # get things ready for next loop
                if done:
                    curr_state = self.memory.flatten_state( self.env.reset() )
                    scores.append(0)
                else:
                    curr_state = next_state

            assert type(curr_state) is torch.Tensor
            return curr_state, scores


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
    runner.n_step_rollout( num_iter=1000, render=True)
    distances, weight_literal, probs, agent_positions, hazard_positions, actions = runner.extract_distances()
    print("State Values:", next_observations)
    print("State Values:", np.array([ flatten_state(x) for x in next_observations ]) )
    print("reward we want:", np.array([ rewards[i]*distances[i] for i in range(len(rewards)) ]).flatten()[:10] )
    # print("Running get_distance() completed successfully")
    # #print("Distances to objects:", distances, "Weight of Literals:", weights, "State observation values:", state_value, "Agent Position:", agent_position)