""" Initialize training for safety-gym or cartpole
"""

from collections import defaultdict
import csv

# pylint: disable=import-error
import numpy as np

from memory import map_and_flatten_state
from runner import Runner

from agents import Agent
from params import Params

# pylint: disable=wildcard-import, unused-wildcard-import
from constraints import *

def default_scores_dict():
    return {"t": [], "episode":[], "val": [], "mean": [], "std": []}

def trainer( env,
        agent: Agent,
        params: Params,
        run_name: str,
        render: bool = False
        ):
    runner = Runner( env, agent )
    curr_state, _state_map = map_and_flatten_state( env.reset() )

    episode, timesteps = 0, 0
    train_run_data = defaultdict(default_scores_dict)
    test_run_data  = defaultdict(default_scores_dict)
    train_states = []
    test_states  = []
    run_data = None

    # generate titles for states and constraints csv
    env.reset()
    action = env.action_space.sample()
    [ state, _reward, _done, info ] = env.step(np.array( action ))

    # TODO: Add real logging library
    # Map names of state variables to corresponding part of array, and vice versa
    curr_state, state_mapping = map_and_flatten_state( state )
    reverse_mapping = ["" for _ in range(params.state_size)]
    for key, [start,end] in state_mapping.items():
        for i in range(start, end):
            reverse_mapping[i] = key+"_"+str(i-start)

    # Define mapping on info dict, action indices, and other titles to save
    info_mapping    = list( info.keys() )
    action_mapping = ["action"]
    if params.actions_continuous:
        action_mapping = [ "action_"+str(i) for i in range(params.action_size) ]
    titles = ["time", "done", "reward", "constraint",
        *action_mapping, *info_mapping, *reverse_mapping ]

    train_states.append( titles )
    test_states.append( titles )
    for state_data, name in [(train_states, "training"), (test_states, "test")]:
        if name == "test" and not params.run_tests:
            continue
        # save the states from this run
        with open(f"runs/{name}-states-{run_name}.csv", "a") as f:
            writer = csv.writer(f)
            for state in state_data:
                writer.writerow(state)

    # Save parameters
    params._dump( f"runs/params-{run_name}.json" )

    train_states, test_states = [], []

    # pre-test agent as a zero datapoint
    # TODO: Re-add code for testing agent

    while timesteps < params.num_timesteps:
        print("timesteps:", timesteps)
        # Step 1. n-step rollout
        curr_state, run_data, state_data = runner.n_step_rollout( agent.choose_action,
            curr_state, params.num_iter, render=render, prev_run_data=run_data,
            training=True, current_time=timesteps )
        train_states.extend( state_data )
        timesteps += params.num_iter

        # step 2. get some info
        num_dones = np.sum( agent.memory.dones )

        # Step 3. Learn
        losses = agent.learn()
        losses = { k:round(float(v),3) for k,v in losses.items()}

        # Step 4. Print some info about the training so far
        initial_episode = episode
        num_dones = len( run_data["score"] ) - 1
        episode += num_dones

        for metric, data in run_data.items():
            for i, value in enumerate(data):
                train_run_data[metric]["val"].append( value )
                train_run_data[metric]["episode"].append( initial_episode + i )

        scores = run_data["score"]

        scores_mean = np.mean( scores[:-1] )
        scores_std  = np.std(  scores[:-1] )
        print(f'# Episodes {initial_episode}-{initial_episode+len(scores)-2}:')
        print(f'{"%20s"%"Score"} = {"%.2f" % scores_mean} ± {"%.2f" % scores_std}')

        for k, v in losses.items():
            print(f'{"%20s"%f"{k}"} = {v}')

        # Step 5. Test the model to gain insight into performance
        # TODO: Re-add code for testing the model

        # Step 6. Save the model
        if params.save_period and timesteps % params.save_period == 0:
            agent.save_models()

        # Step 7. Save logs about the Model states
        for state_data, name in [(train_states, "training"), (test_states, "test")]:
            if name == "test" and not params.run_tests:
                continue
            # save the states from this run
            with open(f"runs/{name}-states-{run_name}.csv", "a") as f:
                writer = csv.writer(f)
                for state in state_data:
                    writer.writerow(state)

        train_states, test_states  = [], []

    print("Finished training")
    return train_run_data, test_run_data
