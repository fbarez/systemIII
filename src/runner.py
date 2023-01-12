""" Define a function which rolls out steps in the environment,
and saves the relevant environment variables
"""
from collections import defaultdict
from typing import Optional, Dict, Callable
import warnings

import torch
import numpy as np
from memory import Memory

from tqdm import tqdm

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
        print("# Rolling out agent in environment")
        for time in tqdm(range(num_iter)):
            # Get the next state
            with torch.no_grad():
                action, action_logprob, action_mean = action_sampler(curr_state, training)

            [ next_state, reward, done, info ] = self.env.step(np.array(action.cpu()))
            curr_data = {"score": reward, **info}

            # Save additional state data for later training
            next_state = memory.flatten_state(next_state)

            # define some some _variables, indicating they might change
            _state, _reward = curr_state, reward
            _pred_state, _value, _cost, _cost_value = zero(), zero(), zero(), zero()

            with torch.no_grad():

                if agent.has_predictor:
                    # THIS NEEDS TO STAY curr_state OR IT DOES NOT WORK ???
                    _state = curr_state
                    # _pred_state = agent.predictor(curr_state, action)
                    # _state = _pred_state

                if agent.has_value_critic:
                    _value = agent.value_critic(_state)

            # Store the transition. Let cost be zero for now, as it is calculated next
            memory.add(curr_state, next_state, _pred_state, action_mean, action,
                action_logprob, reward, _value, _cost, _cost_value, done, info)

            # Calculate constraint and add to memory after (memory used to calculate)
            try:
                with torch.no_grad():
                    cost = self.agent.calculate_constraint(time, next_state, memory)

                if agent.has_cost_critic:
                    _cost_value = agent.cost_critic(_state)

                if agent.params.reward_penalized:
                    curr_penalty = agent.penalty.use_penalty()
                    reward_total = reward - curr_penalty * _cost
                    reward_total = reward_total / (1 + curr_penalty)
                    _reward, _cost, _cost_value  = reward_total, zero(), zero()

                memory.rewards[-1]     = _reward
                memory.costs[-1]       = _cost
                memory.cost_values[-1] = _cost_value

            except NotImplementedError:
                print("Warning: Not implemented")

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

        # Get ready to return things
        actions = torch.stack([action]) if not isinstance(action, list) else action
        info_values = list( info.values() )
        state_data.append([ current_time+time, done, reward, float(cost),
            *actions.cpu().numpy().flatten(), *info_values, *next_state.cpu().numpy() ])

        assert isinstance(curr_state, torch.Tensor)

        return curr_state, run_data, state_data