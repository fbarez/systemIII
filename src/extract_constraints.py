from re import A
import pandas as pd
import numpy as np
import csv
import argparse

def extract_constraints( input_file, output_file=None ):
    fields = ['constraint']

    df = pd.read_csv(input_file, usecols=fields)
    constraints = df['constraint'].to_numpy()

    constraints = 1000 - np.reshape( constraints, (-1, 1000) ).sum( axis=1 )

    data_dict = {
        'episode': np.arange( len(constraints) ),
        'val': constraints,
        't': [],
        'mean': [],
        'std': []
    }
    

    # save constraits as a row in a csv
    if output_file:
        with open(output_file, "w") as f:
            writer = csv.writer(f)
            for k, v in data_dict.items():
                writer.writerow([k, *v])
    
    return data_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot diagrams.')
    parser.add_argument('input_file', metavar='input_file', type=str, help='choose which file to plot')
    parser.add_argument('output_file', metavar='output_file', type=str, help='choose which file to plot')
    args = parser.parse_args()

    extract_constraints( args.input_file, args.output_file )