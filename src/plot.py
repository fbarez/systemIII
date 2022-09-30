from re import I
import matplotlib.pyplot as plt
import numpy as np
import pandas as p
import csv

import argparse

y = [1.409, -3.339, 2.995, 4.501, -2.384, -3.68, 0.165, -6.371, -0.085, -0.535, 3.432, 2.613, 2.433, 0.497, -0.565, 0.787, 4.345, 4.508, 3.275, 0.713, 2.278, 8.532, -0.075, -0.545, 0.271, 6.976, 4.083, 1.183, 1.718, 1.808, 0.738, 3.265, -0.911, 5.132, 4.784, 2.695, 2.422, 2.301, 5.361, 2.753, 2.958, 2.159, 5.806, 2.103, 3.561, 3.009, 5.862, 0.005, 3.202, 2.978, 6.512, 1.703, 3.282, 4.763, 5.199, 7.919, 4.707, 4.635, 2.951, 6.706, 2.065, 7.009, 3.518, 5.727, 3.031, 5.506, 5.032, 2.679, -5.364, -0.701, 3.804, 3.645, 0.452, 6.205, 0.041, -4.767, 0.386, 0.109, 6.815, 1.046, 2.753, 0.589, 3.662, 1.517, 3.277, 3.701, 2.224, 0.702, 5.401, 5.465, 1.888, 4.136, 2.555, 1.39, 0.756, 4.653, 4.265, 6.043, 5.382, 2.875, 5.357, 3.273, 0.628, 5.078, 1.665, 3.233, 4.906, 2.634, 2.005, 2.782, 5.18, 2.692, 0.183, 0.756, 3.224, 7.188, 3.155, 0.453, 1.088, 3.642, 6.801, 3.492, 2.126, 4.567, 3.784, 0.158, 2.685, 4.793, 4.468, 4.482, 6.895, 5.109, 7.047, 4.256, 4.454, 6.112, 4.502, 8.04, 3.688, 5.028, 1.051, 1.182, 4.194, 1.457, 4.673, 4.008, 0.72, 3.401, 2.179, 2.908, 2.717, 0.961, 0.488, 2.534, 1.534, 2.284, 2.26, 0.141, 4.201, 4.188, -22.389, -0.665, -4.893, -1.561, -7.554, -6.376, 0.505, 2.172, 0.724, 0.884, 1.022, 0.951, 4.859, 1.557, 1.782, 3.603, 0.507, 0.76, 3.503, 2.471, 1.492, 1.414, 1.572, 0.505, 4.244, 0.753, -0.159, 1.299, 2.209, 0.844, 2.061, 0.975, 2.238, 1.643, 0.827, 1.49, 1.613, 1.392, 0.246, 0.231, 1.447, 1.048, 1.438, 2.585, 0.703, 2.138, 1.132, 0.267, 1.706, 1.292, 1.038, 2.597, 2.578, 0.671, 1.701, 0.405, 0.986, 2.156, 1.292, -0.102, 3.104, 0.865, 0.725, 1.848, 2.553, 2.735, 1.608, 2.659, 2.663, 1.293, 3.25, 3.074, 2.513, 0.403, 2.746, 0.894, 2.438, 2.22, 0.597, 1.537, 0.532, 2.196, 4.622, 1.192, 0.708, 2.277, 3.601, 2.169, 1.077, 3.514, 2.838, 4.766, 2.026, 1.482, 2.974, 1.442, 1.408, 2.948, 2.951, 3.293, 3.036, 1.966, 4.195, 4.5, 1.646, 1.984, 3.872, 2.829, 3.121, 1.754, 2.721, 2.783, 3.855, 1.723, 2.875, 2.857, 3.168, 1.539, 3.476, 1.278, 2.749, 2.977, 3.868, 1.287, 1.686, 3.127, 3.961, 4.097, 1.149, 4.459, 5.464, 3.915, 3.054, 4.511, 3.561, 4.889, 1.5, 3.935, 2.008, 0.892, 1.771]

def plot_scores(data, window_size=10, sigmas=2):
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
        y_mean = y_rolling.mean()
        
        # generate the standard deviation of the rolling average windows:
        y_std = y_rolling.std()

        rolling = True
    
    else:
        raise ValueError("Data must have either a 'mean' or 'val' column")
    
    if len(data["t"]):
        x, x_label = np.array(data["t"]), "Timesteps"
    elif len(data["episode"]):
        x, x_label = np.array(data["episode"]), "Episodes"
    else:
        x = range(len(y))

    # generate the upper and lower bounds of the rolling average windows:
    y_upper = y_mean + (y_std * sigmas)
    y_lower = y_mean - (y_std * sigmas)

    # Draw plot with error band and extra formatting to match seaborn style
    fig, ax = plt.subplots(figsize=(9,5))
    if y:
        ax.plot(x, y, label='score', color='tab:purple', alpha=0.1)
    ax.plot(x, y_mean, label='score mean')
    ax.plot(x, y_lower, color='tab:blue', alpha=0.1)
    ax.plot(x, y_upper, color='tab:blue', alpha=0.1)
    ax.fill_between(x, y_lower, y_upper, alpha=0.2)
    ax.set_xlabel(x_label)
    if rolling:
        ax.set_ylabel('rolling mean score')
    else:
        ax.set_ylabel('mean score')
    ax.spines['top'  ].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig

if __name__ == '__main__':
    # load data file
    parser = argparse.ArgumentParser(description='Plot diagrams.')
    parser.add_argument('filenames', metavar='filenames', type=str, nargs='+',
                    help='choose which file to plot')
    parser.add_argument('--window_size', type=int, default=10)
    args = parser.parse_args()
    window_size = args.window_size

    for filename in args.filenames:
        with open(filename, "r") as f:
            reader = csv.reader(f)
            # each row has the name of the data in the first column
            # and the data in the next columns
            data = { row[0]:row[1:] for row in reader }


            scores = list(reader)[0]
            scores = [float(score) for score in scores]
            fig = plot_scores(scores, window_size=window_size, sigmas=1)

    plt.show()