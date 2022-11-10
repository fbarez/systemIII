from re import A
import pandas as pd
import numpy as np
import csv
import argparse

def extract_from_states( input_file, output_file=None, header='constraint' ):
    fields = [ header ]

    df = pd.read_csv(input_file, usecols=fields)
    values = df[ header ].to_numpy()

    values = np.reshape( values, (-1, 1000) ).sum( axis=1 )

    data_dict = {
        'episode': np.arange( len(values) ),
        'val': values,
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
    parser = argparse.ArgumentParser(description='Extract a column of data from the states CSV and summarise into episodes')
    parser.add_argument('input_file', metavar='input_file', type=str, help='input file of training_states')
    parser.add_argument('output_file', metavar='output_file', type=str, help='output file of training-scores.csv')
    parser.add_argument('--header', metavar='output_file', type=str, help='header to look at', default='constraints')
    args = parser.parse_args()

    extract_from_states( args.input_file, args.output_file, args.header )