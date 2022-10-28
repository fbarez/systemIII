import numpy as np
import csv
import argparse

# change data from openai safefy starter agents into different format used by this repo
def convert_scores_format( input_filename, output_filename, mode='reward' ):
    mean_map = { 'reward': 'AverageEpRet', 'cost': 'AverageEpCost' }
    std_map = { 'reward': 'StdEpRet', 'cost': 'StdEpCost' }
    mean = mean_map[mode]
    std = std_map[mode]
    with open( input_filename, "r" ) as f:
        reader = csv.reader( f, delimiter="\t" )
        # Read csv file . First line has titles, other lines have data
        titles = next(reader)
        print( titles )
        original_data = { title:[] for title in titles }
        for row in reader:
            for title, value in zip(titles, row):
                original_data[title].append( float(value) )
        
        data = {'t': [], 'epidose': [], 'val': [], 'mean': [], 'std': []}
        data['t'] = ( np.array( original_data['Epoch'], dtype=np.int64 )+1 )*30000
        data['episode'] = ( np.array( original_data['Epoch'], dtype=np.int64 )+1 )*30
        data['mean']    = np.array( original_data[mean], dtype=np.float32 )
        data['std']     = np.array( original_data[std], dtype=np.float32 )
        print( data )

    # Save to csv file, with title in the first column
    with open( output_filename, "w" ) as f:
        writer = csv.writer( f, delimiter="\t" )
        for key, value in data.items():
            writer.writerow( [key] + list(value) )

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert data from openai safety starter agents to data format used by this repo.')
    parser.add_argument( 'input_filename', metavar='input_filename', type=str )
    parser.add_argument( 'output_filename', metavar='output_filename', type=str )
    parser.add_argument( '--mode', type=str, default='reward' )
    args = parser.parse_args()

    print( convert_scores_format( args.input_filename, args.output_filename, args.mode ) )
