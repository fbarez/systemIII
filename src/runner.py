""" Define a function which rolls out steps in the environment,
and saves the relevant environment variables
"""
from collections import defaultdict
from typing import Optional, Dict, Callable
import warnings

import torch
import numpy as np
from memory import Memory

warnings.filterwarnings('ignore')
print("ignore...\n")


def zero():
    return torch.tensor(0)

# , 'Position Agent', next_observation['observe_qpos']
class Runner(object):
    """ Rolls out steps in the environment.
    """
    def __init__(self, env, agent=None):
        self.env = env
        self.agent = agent
        self.memory = Memory() if agent is None else self.agent.memory
        return

    def random_action_sampler( self, _state ):
        action = self.env.action_space.sample()
        return action, 0, zero()

    def n_step_rollout( self,
                        action_sampler: Optional[Callable] = None,
                        curr_state: Optional[torch.Tensor] = None,
                        num_iter: int = 100,
                        render: bool  = False,
                        prev_run_data: Optional[Dict[str, list]] = None,
                        training: bool = False,
                        current_time: int = 0 ):
        """[summary]
        action_sampler: function that takes in a state and returns an action
        current_state: the initial state of the rollout
        num_iter: number of iterations to run the rollout
        render: whether to render the environment
        """
        self.memory = self.memory if self.agent is None else self.agent.memory
        memory = self.memory
        agent = self.agent

        if prev_run_data is None:
            run_data = defaultdict(list)
        else:
            run_data = defaultdict(list, prev_run_data)
            for k, v in prev_run_data.items():
                run_data[k] = v[-1:]

        def add_to_run_data( key, value, done ):
            if key not in run_data:
                run_data[key] = [0]
            run_data[key][-1] += value
            if done:
                run_data[key].append(0)

        # if no agent provided, use a random action sampler
        if action_sampler is None:
            action_sampler = self.random_action_sampler

        # initialise state
        if curr_state is None:
            curr_state = memory.flatten_state( self.env.reset() )

        state_data = []
        for i in range(num_iter):
            # Get the next state

            with torch.no_grad():
                action, action_logprob, action_mean = action_sampler(curr_state, training)

            [ next_state, reward, done, info ] = self.env.step(np.array(action.cpu()))
            curr_data = {"score": reward, **info}

            # Save additional state data for later training
            next_state = memory.flatten_state(next_state)
            # pylint

            with torch.no_grad():
                pred_state = agent.run_if_has('predictor',state=curr_state, action=action)
                value      = agent.run_if_has('value_critic', state=next_state)

            # Store the transition. Let cost be zero for now, as it is calculated next
            _cost, _cost_value = zero(), zero()
            memory.add(curr_state, next_state, pred_state, action_mean, action,
                action_logprob, reward, value, _cost, _cost_value, done, info)

            # Calculate constraint and add to memory after (memory used to calculate)
            try:
                with torch.no_grad():
                    cost = self.agent.calculate_constraint(i, next_state, memory)
                    cost_value = agent.run_if_has('cost_critic',  state=next_state)
                memory.costs[-1] = cost_value

                # Save cost differently depending on if reward is penalized
                _reward, _cost, _cost_value = reward, cost, cost_value

                if agent.params.reward_penalized:
                    curr_penalty = agent.penalty.use_penalty()
                    reward_total = reward - curr_penalty * cost
                    reward_total = reward_total / (1 + curr_penalty)
                    _reward, _cost, _cost_value  = reward_total, zero(), zero()

                memory.rewards[-1]     = _reward
                memory.costs[-1]       = _cost
                memory.cost_values[-1] = _cost_value

            except NotImplementedError:
                pass

            # Save current state data
            for key, value in curr_data.items():
                add_to_run_data(key, value, done)

            if render:
                self.env.render()

            # get things ready for next loop
            if done:
                curr_state = memory.flatten_state( self.env.reset() )
            else:
                curr_state = next_state

            actions = torch.stack([action]) if not isinstance(action, list) else action
            info_values = list( info.values() )
            state_data.append([ current_time+i, done, reward, float(cost),
              *actions.cpu().numpy().flatten(), *info_values, *next_state.cpu().numpy() ])

            assert isinstance(curr_state, torch.Tensor)
            return curr_state, run_data, state_data

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
