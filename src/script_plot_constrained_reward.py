from inspect import Parameter
import re
import pandas as pd
import numpy as np
import csv
import argparse
from agents import S3Agent
from params import Params
import torch
from torch import tensor as t
import matplotlib.pyplot as plt

def extract_from_states( input_file, output_file ):
    fields = [ 'time', 'done', 'reward', 'cost', 'constraint']

    df = pd.read_csv(input_file, usecols=fields, skiprows=range(1,int(3e6)), nrows=1e4)
    times       = df[ 'time' ].to_numpy()
    dones       = df[ 'done' ].to_numpy()
    rewards     = df[ 'reward' ].to_numpy()
    costs       = df[ 'cost' ].to_numpy()
    constraints = df[ 'constraint' ].to_numpy()

    s3 = S3Agent
    s3.reward_decay = 0.99
    s3.gae_lambda = 0.95
    s3.device = 'cpu'
    cumulative_rewards  = s3.calculate_cumulative_rewards( s3, t(rewards), t(dones) )
    cumulative_cost  = s3.calculate_cumulative_rewards( s3, t(costs), t(dones) )

    constrained_rewards = s3.calculate_constrained_rewards( s3, cumulative_rewards, t(constraints) )
    fake_constrained_rewards = s3.calculate_constrained_rewards( s3, cumulative_rewards, 1-t(costs) )

    s3.reward_decay = 0.98
    cumulative_inv_constraints = s3.calculate_cumulative_rewards( s3, 1-constrained_rewards, t(dones) )

    fig, axes = plt.subplots(figsize=(15, 8), nrows=3, ncols=1)

    # plot constraints
    ax = axes[0]
    ax.plot( times, 1-costs,     linewidth=1, label='cost constraints')
    ax.plot( times, constraints, linewidth=1, label='computed constraints')
    ax.legend(loc='best')

    # plot time reward
    ax = axes[1]
    ax.hlines(y=0, xmin=times[0], xmax=times[-1], linewidth=0.5, linestyle='dashed', color='black')
    ax.plot( times, rewards, linewidth=1, label='constrained reward', color='red')
    ax.set_ylim([-0.05, 0.05])
    ax.legend(loc='best')

    # plot cumulative reward
    ax = axes[2]
    #ax.plot( times, cumulative_rewards - cumulative_cost*0.06, color='black', linewidth=1, label='negative cost term')
    #ax.plot( times, cumulative_rewards - cumulative_inv_constraints*0.03, color='darkblue', linewidth=1, label='calculated negative cost term')
    ax.plot( times, cumulative_rewards, color='green', linewidth=1, label='cumulative reward')
    ax.plot( times, constrained_rewards, linewidth=1, label='real constrained reward')
    ax.plot( times, fake_constrained_rewards, linewidth=1, label='cost constrained reward')

    ax.legend(loc='best')

    plt.savefig( output_file, dpi=300 )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract a column of data from the states CSV and summarise into episodes')
    parser.add_argument('input_file', metavar='input_file', type=str, help='input file of training_states.csv')
    parser.add_argument('output_file', metavar='output_file', type=str, help='output file of plot.png')
    args = parser.parse_args()

    extract_from_states( args.input_file, args.output_file )