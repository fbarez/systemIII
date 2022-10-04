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

def plot_scores(data, window_size:int=10, sigmas:float=2, fig:Optional[Figure]=None, color:str="blue", label="Mean Score"):
    if type(data) is not dict:
        data = {"val": data}

    if len(data["mean"]) != 0:
        y_mean, y_std = np.array(data["mean"]), np.array(data["std"])
        y = np.array(data["val"]) if len(data["val"]) != 0 else None
        rolling = False

        y_mean = p.Series(y_mean).rolling(window=window_size, min_periods=1).mean()
        y_std  = p.Series(y_std ).rolling(window=window_size, min_periods=1).mean()
    
    elif len(data["val"]) != 0:
        # generate the rolling average of x:
        min_periods = np.min([window_size, 1])
        y = data["val"]
        y_rolling = p.Series(y).rolling(window=window_size, min_periods=min_periods)
        y_mean = y_rolling.mean()
        
        # generate the standard deviation of the rolling average windows:
        y_std = y_rolling.std()

        data["val"]  = np.array(y)
        data["mean"] = np.array(y_mean)
        data["std"]  = np.array(y_std)

        rolling = True
    
    else:
        raise ValueError("Data must have either a 'mean' or 'val' column")
    
    if len(data["t"]):
        x, x_label = np.array(data["t"]), "Timesteps"

    elif len(data["episode"]):
        x, x_label = np.array(data["episode"]), "Episodes"
    else:
        x = np.array(list(range(len(y))))
        data["t"] = x

    # generate the upper and lower bounds of the rolling average windows:
    y_upper = y_mean + (y_std * sigmas)
    y_lower = y_mean - (y_std * sigmas)

    # Draw plot with error band and extra formatting to match seaborn style
    if fig:
        ax = fig.axes[0]
    else:
        fig, ax = plt.subplots(figsize=(9,5))
    if y:
        ax.plot(x, y, label='score', color='tab:purple', alpha=0.1)
    ax.plot(x, y_mean, label=label, color=f'tab:{color}')
    ax.plot(x, y_lower, color=f'tab:{color}', alpha=0.1)
    ax.plot(x, y_upper, color=f'tab:{color}', alpha=0.1)
    ax.fill_between(x, y_lower, y_upper, alpha=0.2, color=f'tab:{color}')
    ax.set_xlabel(x_label)
    if rolling:
        ax.set_ylabel('Rolling Mean Score')
    else:
        ax.set_ylabel('Return')
    ax.spines['top'  ].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig, data

if __name__ == '__main__':
    # load data file
    parser = argparse.ArgumentParser(description='Plot diagrams.')
    parser.add_argument('filenames', metavar='filenames', type=str, nargs='+',
                    help='choose which file to plot')
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()
    window_size = args.window_size

    all_data = defaultdict(list)
    for filename in args.filenames:
        with open(filename, "r") as f:
            agent_type = 'ppo'
            if 's3' in filename:
                agent_type = 's3'
            if 'ac' in filename:
                agent_type = 'ac'
            
            reader = csv.reader(f)
            # each row has the name of the data in the first column
            # and the data in the next columns
            data = { row[0]:[ float(i) for i in row[1:]] for row in reader }
            if args.verbose:
                fig, data = plot_scores(data, window_size=window_size, sigmas=1)
                fig.canvas.manager.set_window_title(filename)
            data["episode"] = np.array(data["episode"])
            print(data)
            all_data[agent_type].append(data)

    # plot the mean of all data
    fig, ax = plt.subplots(figsize=(9,5))
    color_map = {'ppo': 'blue', 's3': 'orange', 'ac': 'green'}
    label_map = {'ppo': 'Proximal Policy Optimization', 's3': 'System 3', 'ac': 'Proximal Policy Optimization'}
    for key, data_list in all_data.items():
        if not len( data_list[0]["mean"] ):
            continue
        test_scores_dict = {
            "mean": np.mean( [ d["mean"] for d in data_list ], axis=0 ),
            "std":  np.std(  [ d["mean"] for d in data_list ], axis=0 ),
            "t":    np.mean( [ d["t"]    for d in data_list ], axis=0 ),
            "val":  [],
            "episode": np.mean([ d["episode"] for d in data_list ], axis=0 ),
        }
        # plot the scores
        print( data_list, "\n\ntest scores:", test_scores_dict )
        plot_scores( test_scores_dict, window_size=20, sigmas=1, fig=fig, color=color_map[key], label=label_map[key] )

    plt.legend()
    plt.show()