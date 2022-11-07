from re import I
import matplotlib.pyplot as plt
import numpy as np
import pandas as p
import csv
import re
from collections import defaultdict
import argparse
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from typing import Optional
import seaborn as sns
sns.set_theme()

# Hardcoded values
plt.rcParams["figure.figsize"] = (9, 5)
XMIN, XMAX = 0, 1e7
LINE_WIDTH = 3
EP_PER_EPOCH = 30

def plot_scores(data,
        window_size: int = 10,
        sigmas: float = 1,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        color: str = "blue",
        label: str = "Mean Score",
        min_periods: int = 1,
        yscale: float = 1,
        ):
    if type(data) is not dict:
        data = {"val": data}

    min_periods = int( np.median([1, min_periods, window_size]) )

    if len(data["mean"]) != 0:
        y_mean, y_std = np.array(data["mean"]), np.array(data["std"])
        y = np.array(data["val"]) if len(data["val"]) != 0 else None
        rolling = False

        y_mean = p.Series(y_mean).rolling(
            window=window_size, min_periods=min_periods).mean()
        y_std  = p.Series(y_std ).rolling(
            window=window_size, min_periods=min_periods).mean()
    
    elif len(data["val"]) != 0:
        # generate the rolling average of x:
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

    # scale axes
    y_mean *= yscale
    y_std  *= yscale
    y = None if y is None else y*yscale

    # generate the upper and lower bounds of the rolling average windows:
    y_upper = y_mean + (y_std * sigmas)
    y_lower = y_mean - (y_std * sigmas)

    # Draw plot with error band and extra formatting to match seaborn style
    if not fig is None:
        ax = fig.axes[0]
    elif ax is None:
        fig, ax = plt.subplots()

    # Plot seaborn line and the error bands
    ax.plot(x, y_mean, label=label, color=color, linewidth=LINE_WIDTH)
    ax.fill_between(x, y_lower, y_upper, alpha=0.1, color=color)
    ax.set_xlabel(x_label)
    if rolling:
        ax.set_ylabel(f'Rolling Mean {label}')
    else:
        ax.set_ylabel(label)
    plt.xlim(xmin=XMIN, xmax=XMAX)
    
    if y:
        ax.plot(x, y, label='raw data', color='purple', alpha=0.1)

    return ax, data

if __name__ == '__main__':
    # load data file
    parser = argparse.ArgumentParser(description='Plot diagrams.')
    parser.add_argument('filenames', metavar='filenames', type=str, nargs='+',
                    help='choose which file to plot')
    parser.add_argument('-w', '--window_size', type=int, default=10)
    parser.add_argument('--min_periods', type=float, default=1)
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('-l', '--ylabel', type=str, default='AverageEpRet')
    parser.add_argument('--yscale', type=float, default=1)
    parser.add_argument('--compare_line', type=float, default=None)
    parser.add_argument('--num_sigmas', type=float, default=1)
    parser.add_argument('--sum_mode', action='store_true', default=False )
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    args = parser.parse_args()
    window_size = args.window_size

    all_data = defaultdict(list)
    for filename in args.filenames:
        with open(filename, "r") as f:
            agent_type = 'ppo'
            if 'ppo_lagrangian' in filename:
                agent_type = 'ppo_lagrangian'
            elif 'trpo_lagrangian' in filename:
                agent_type = 'trpo_lagrangian'
            elif 'ppo' in filename:
                continue
                agent_type = 'ppo'
            elif 'trpo' in filename:
                agent_type = 'trpo'
            elif 'cpo' in filename:
                agent_type = 'cpo'
            elif 's3' in filename:
                agent_type = 's3'
            elif 'ac' in filename:
                agent_type = 'ac'
            
            reader = csv.reader(f)
            # each row has the name of the data in the first column
            # and the data in the next columns
            data = { row[0]:[ float(i) for i in row[1:]] for row in reader }
            if args.verbose:
                ax, data = plot_scores(data, window_size=window_size,
                    min_periods=args.min_periods, yscale=args.yscale )
                ax.figure.canvas.manager.set_window_title(filename)
            data["episode"] = np.array(data["episode"])
            if len(data['t']) == 0:
                data['t'] = data['episode']*1000
            all_data[agent_type].append(data)


    # plot the mean of all data
    fig, ax = plt.subplots()
    plt.hlines( args.compare_line, 0, 1e7, color='darkred',
                linestyle='dashed', linewidth=LINE_WIDTH )

    color_map = {
        "ppo":             "tab:green",
        'ac':              'tab:green',
        "ppo_lagrangian":  "tab:red",
        "trpo":            "tab:purple",
        "trpo_lagrangian": "darkgoldenrod",
        "cpo":             "tab:blue",
        's3': 'orange',
    }
    label_map = {
        's3':              'System 3',
        'ac':              'PPO',
        "ppo":             "PPO",
        "ppo_lagrangian":  "PPO Lagrangian",
        "trpo":            "TRPO",
        "trpo_lagrangian": "TRPO Lagrangian",
        "cpo":             "CPO"
    }
    for key, data_list in all_data.items():
        print( key, len(data_list) )
        if not len( data_list[0]["mean"] ):
            continue
        test_scores_dict = {
            "mean": np.mean( [ d["mean"] for d in data_list ], axis=0 ),
            "std":  np.std(  [ d["mean"] for d in data_list ], axis=0 ),
            "t":    np.mean( [ d["t"]    for d in data_list ], axis=0 ),
            "val":  [],
            "episode": np.mean([ d["episode"] for d in data_list ], axis=0 ),
        }

        # Sum mode shows cumulative costs etc
        if args.sum_mode:
            means = test_scores_dict["mean"] * EP_PER_EPOCH
            stds  = test_scores_dict["std"]  * EP_PER_EPOCH
            for i in range(1, len(means)):
                means[ i ] = means[i] + means[i-1]
                stds[ i ]  = stds[i] + stds[i-1]
                test_scores_dict["mean"] = means
                test_scores_dict["std"]  = stds
                window_size = 1
        
        if len(data_list) == 1:
            test_scores_dict["std"] = data_list[0]["std"]

        # plot the scores
        color, label = color_map[key], label_map[key]
        ax, data = plot_scores( test_scores_dict, window_size=window_size,
            sigmas=args.num_sigmas, yscale=args.yscale, ax=ax, color=color,
            label=label, min_periods=args.min_periods )

    if args.compare_line:
        print( args.compare_line )

    plt.ylabel( args.ylabel )
    plt.legend()
    plt.savefig( args.output, dpi=300 )
    plt.show()