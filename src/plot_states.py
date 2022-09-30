from re import I
import matplotlib.pyplot as plt
import numpy as np
import pandas as p
import csv
import re
from collections import defaultdict
import argparse
from matplotlib.figure import Figure
from typing import Optional

def constraint_accumulation(data, window_size:int=2048, threshold:float=1):
    y = np.max([ threshold - np.array(data["val"]), np.zeros_like(data["val"]) ], axis=0)
    x = np.array(data["t"])

    y_plot = []

    i = 0
    while i < len(y):
        y_plot.append(np.sum(y[i:i+window_size]))
        i += window_size
    
    x_plot = x[::window_size]*18

    return {
        "t": x_plot,
        "mean": y_plot,
        "std": np.zeros(len(y_plot)),
        "episode": [],
        "val": []
    }


def plot_states(data, window_size:int=10, sigmas:float=2, fig:Optional[Figure]=None, color:str="blue", label="Mean Score"):
    if type(data) is not dict:
        data = {"val": data}

    if len(data["mean"]) != 0:
        y_mean, y_std = np.array(data["mean"]), np.array(data["std"])
        y = np.array(data["val"]) if len(data["val"]) != 0 else None
        rolling = False
    
    elif len(data["val"]) != 0:
        # generate the rolling average of x:
        min_periods = np.min([window_size, 10])
        y = data["val"]
        y_rolling = p.Series(y).rolling(window=window_size, min_periods=min_periods)
        y_mean = y_rolling.min()
        
        # generate the standard deviation of the rolling average windows:
        y_std = y_rolling.std()

        rolling = True
    
    else:
        raise ValueError("Data must have either a 'mean' or 'val' column")
    
    if len(data["t"]):
        x, x_label = np.array(data["t"]), "Timesteps"
        x, x_label = x/10000, "Episodes"

    elif len(data["episode"]):
        x, x_label = np.array(data["episode"]), "Episodes"
    else:
        x = range(len(y))

    # generate the upper and lower bounds of the rolling average windows:
    y_upper = y_mean + (y_std * sigmas)
    y_lower = y_mean - (y_std * sigmas)
    y_lower = np.max([y_lower, np.zeros_like(y_lower)], axis=0)

    # Draw plot with error band and extra formatting to match seaborn style
    if fig:
        ax = fig.axes[0]
    else:
        fig, ax = plt.subplots(figsize=(9,5))
    #if y:
    #    ax.plot(x, y, label='score', color='tab:purple', alpha=0.1)
    ax.plot(x, y_mean, label=label, color=f'tab:{color}')
    ax.plot(x, y_lower, color=f'tab:{color}', alpha=0.1)
    ax.plot(x, y_upper, color=f'tab:{color}', alpha=0.1)
    ax.fill_between(x, y_lower, y_upper, alpha=0.2, color=f'tab:{color}')
    ax.set_xlabel(x_label)
    if rolling:
        ax.set_ylabel('Rolling Mean Score')
    else:
        ax.set_ylabel('Constraint Violation Score')
    ax.spines['top'  ].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig

if __name__ == '__main__':
    # load data file
    parser = argparse.ArgumentParser(description='Plot diagrams.')
    parser.add_argument('filenames', metavar='filenames', type=str, nargs='+',
                    help='choose which file to plot')
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()
    window_size = args.window_size

    training_step_size = 2048
    all_data = defaultdict(list)
    color_map = {'ppo': 'blue', 's3': 'orange', 'ac': 'green'}
    label_map = {'ppo': 'Proximal Policy Optimization', 's3': 'System 3', 'ac': 'Proximal Policy Optimization'}
    for filename in args.filenames:
        with open(filename, "r") as f:
            agent_type = 'ppo'
            if 's3' in filename:
                agent_type = 's3'
            if 'ac' in filename:
                agent_type = 'ac'
            
            reader = csv.reader(f)
            
            #first row of csv is the header
            header = next(reader)

            # the rest of the rows containt the data
            data = [row for row in reader]
            print( "\n", filename, ":\n", data[0] )

            index = 3
            constraint_data = {"val":[ float(d[index]) for d in data ], "t":[ float(d) for d in range(len(data)) ], "mean":[], "std":[]}
            # print(data)
            # fig = plot_states(constraints, window_size=200, sigmas=1)
            accum_constraints = constraint_accumulation(constraint_data, window_size=training_step_size, threshold=0.3)
            if args.verbose:
                fig = plot_states(accum_constraints, color=color_map[agent_type], label=label_map[agent_type])
                fig.canvas.manager.set_window_title(filename)
            all_data[agent_type].append( accum_constraints )

    plt.show()

    # plot the mean of all data
    fig, ax = plt.subplots(figsize=(9,5))
    color_map = {'ppo': 'blue', 's3': 'orange', 'ac': 'green'}
    label_map = {'ppo': 'Proximal Policy Optimization', 's3': 'System 3', 'ac': 'Proximal Policy Optimization'}
    for key, data_list in all_data.items():
        test_states_dict = {
            "mean": np.mean( [ d["mean"] for d in data_list ], axis=0 ),
            "std":  np.std(  [ d["mean"] for d in data_list ], axis=0 ),
            "t":    np.mean( [ d["t"]    for d in data_list ], axis=0 ),
            "val":  [],
            "episode": [],
        }
        # plot the scores
        # print( data_list, test_states_dict )
        fig = plot_states( test_states_dict, window_size=10, sigmas=1, fig=fig, color=color_map[key], label=label_map[key] )

    plt.legend()
    plt.show()