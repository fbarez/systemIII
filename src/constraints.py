from agents import Agent
from torch import Tensor
from memory import Memory
from typing import Optional
import torch

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