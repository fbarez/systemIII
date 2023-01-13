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


    # generate titles for states csv
    titles = build_logging_titles( env, params )
    train_states = [ titles ]

    # Save parameters
    params._dump( f"runs/params-{run_name}.json" )

    # Initialize variables
    curr_state, _state_mapping = map_and_flatten_state( env.reset() )
    train_run_data = defaultdict(default_scores_dict)
    episode, timesteps = 0, 0
    run_data = None

    while timesteps < params.num_timesteps:
        # Step 1. n-step rollout
        print("\n# Rolling out agent in environment. (current timesteps:", timesteps, ")")
        curr_state, run_data, state_data = runner.n_step_rollout( agent.choose_action,
            curr_state, params.num_iter, render=render, prev_run_data=run_data,
            training=True, current_time=timesteps )
        train_states.extend( state_data )
        timesteps += params.num_iter

        # Step 2. Learn
        print("# Training model...")
        losses = agent.learn()

        # Step 3. Print some info about the training so far
        print_epoch_data(episode, run_data, train_run_data, losses)

        # Step 4. Save the model
        if params.save_period and timesteps % params.save_period == 0:
            agent.save_models()

        # Step 5. Save logs about the Model states
        with open(f"runs/training-states-{run_name}.csv", "a") as f:
            writer = csv.writer(f)
            for state in train_states:
                writer.writerow(state)

        # Step 6. Keep track of episode and clear states
        episode += len( run_data["score"] ) - 1
        train_states = []

    print("Finished training")
    return train_run_data

def build_logging_titles(env, params):
    # Do an example run on the environment
    _curr_state, state_mapping = map_and_flatten_state( env.reset() )
    action = env.action_space.sample()
    [ _state, _reward, _done, info ] = env.step(np.array( action ))

    # define mapping for action.
    action_mapping = [ "action" ]
    if params.actions_continuous:
        action_mapping = [ "action_"+str(i) for i in range(params.action_size) ]

    # Define mapping on info dict
    info_mapping    = list( info.keys() )

    # Map names of state variables to corresponding part of array, and vice versa
    reverse_mapping = [ "" for _ in range(params.state_size) ]
    for key, [start,end] in state_mapping.items():
        for i in range(start, end):
            reverse_mapping[i] = key+"_"+str(i-start)

    titles = [ "time", "done", "reward", "constraint",
        *action_mapping, *info_mapping, *reverse_mapping ]

    return titles

def print_epoch_data(episode, run_data, train_run_data, losses):
    initial_episode = episode

    for metric, data in run_data.items():
        for i, value in enumerate(data):
            train_run_data[metric]["val"].append( value )
            train_run_data[metric]["episode"].append( initial_episode + i )

    scores = run_data["score"]

    scores_mean = np.mean( scores[:-1] )
    scores_std  = np.std(  scores[:-1] )
    print(f'# Episodes {initial_episode}-{initial_episode+len(scores)-2}:')
    print(f'{"%20s"%"score"} = {"%.2f" % scores_mean} ± {"%.2f" % scores_std}')

    losses = { k:round(float(v),3) for k,v in losses.items()}
    for k, v in losses.items():
        print(f'{"%20s"%f"{k}"} = {v}')
