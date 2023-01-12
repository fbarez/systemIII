
from typing import Optional
import torch
from torch import Tensor
import numpy as np

from agents import Agent
from memory import Memory

def __clamp( val, lower, upper ):
    r = upper - lower
    u = ( torch.clamp( val, min=lower, max=upper ) - lower )*(1/r)
    return u

def __square_clamp( val, lower, upper ):
    u = __clamp( val, lower, upper )
    return u**2

def __inv_square_clamp( val, lower, upper ):
    u = __clamp( val, lower, upper )
    return 1 - (1-u)**2

def calculate_constraint_cartpole( self: Agent,
        index: int,
        state: Tensor,
        memory: Optional[Memory] = None,
        ):
    pole_angle, pole_max = state[2], 0.2095
    hazardous_distance = 0.1
    angle_remaining = pole_max - torch.abs(pole_angle)
    constraint = ( 1 - torch.clamp(angle_remaining, 0, hazardous_distance)*(1/hazardous_distance) )
    return 1 - constraint

def calculate_constraint_cargoal2_v0( self: Agent,
        index: int,
        state: Tensor,
        memory: Optional[Memory] = None
        ):
    memory = memory if (not memory is None) else self.memory
    cost = memory.infos[index]['cost']
    return torch.tensor(cost).to(self.device)

def calculate_constraint_cargoal2_v1( self: Agent,
        index: int,
        state: Tensor,
        memory: Optional[Memory] = None,
        ):
    max_lidar_range = 5
    hazardous_distance = 0.4
    memory = memory if not memory is None else self.memory

    hazards_lidar = memory.flat_get( state, 'hazards_lidar' )
    closest_hazard = ( 1 - torch.max(hazards_lidar) )*max_lidar_range
    hazards_constraint = ( 1 - torch.clamp( closest_hazard, min=0, max=hazardous_distance )*(1/hazardous_distance) )**2

    constraint = hazards_constraint

    return 1 - constraint

def calculate_constraint_cargoal2_v2( self: Agent,
        index: int,
        state: Tensor,
        memory: Optional[Memory] = None,
        ):
    max_lidar_range = 5
    min_dist = self.params.dist_lower_bound
    max_dist = self.params.dist_upper_bound
    memory = memory if not memory is None else self.memory

    hazards_lidar = memory.flat_get( state, 'hazards_lidar' )
    closest_hazard = ( 1 - torch.max(hazards_lidar) )*max_lidar_range
    hazards_constraint = __inv_square_clamp( closest_hazard, min_dist, max_dist )

    vases_lidar = memory.flat_get( state, 'vases_lidar' )
    closest_vase = ( 1 - torch.max(vases_lidar) )*max_lidar_range
    vases_constraint = __inv_square_clamp( closest_vase, min_dist, max_dist )

    constraint = hazards_constraint*vases_constraint

    return constraint

# Moved here from runner.py (written by Fazl)
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

    return distances, weight_literal, probs, \
        agent_positions, hazard_positions, actions