from matplotlib import pyplot as plt
import numpy as np
import csv
import argparse
from plot_scores import plot_scores
import matplotlib.pyplot as plt

# change data from openai safefy starter agents into different format used by this repo
def convert_scores_format( input_filename, output_filename=None, mode='reward' ):
    filename_map = { 'reward': 'training_scores', 'cost': 'training_costs'}
    default_filename = filename_map[mode]

    if output_filename is None:
        if not '/' in input_filename:
            output_filename = f'{default_filename}.csv'
        else:
            input_filename_split = input_filename.split('/')
            output_filename = '/'.join( input_filename_split[:-1] ) + f'/{default_filename}.csv'

    with open( input_filename, "r" ) as f:
        original_data = {}
        data = {'t': [], 'episode': [], 'val': [], 'mean': [], 'std': []}

        reader = csv.reader( f, delimiter=',' )
        for line in reader:
            key, values = line[0], line[1:]
            original_data[key] = np.array( values )

        vals_arr = np.array( original_data['val'], dtype=np.float )
        for i in np.array(original_data['episode'], dtype=np.int64)[::30]:
            vals = np.array( vals_arr[i:i+30] )
            data['mean'].append( np.mean( vals ) )
            data['std'].append( np.std( vals ) )
            data['episode'].append( i )
            data['t'].append( i*1000 )

    # Save to csv file, with title in the first column
    with open( output_filename, "w" ) as f:
        writer = csv.writer( f, delimiter="," )
        for key, value in data.items():
            writer.writerow( [key] + list(value) )

    fig, data = plot_scores( data )
    plt.savefig( output_filename + '.png' )
    plt.close()
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert data from openai safety starter agents to data format used by this repo.')
    parser.add_argument( 'input_filenames', metavar='input_filename', type=str, nargs='+' )
    parser.add_argument( '--output_filename', metavar='output_filename', type=str )
    parser.add_argument( '--mode', type=str, default='reward' )
    args = parser.parse_args()

    if len( args.input_filenames ) == 1:
        convert_scores_format( args.input_filenames[0], args.output_filename, args.mode )
    
    else:
        for input_filename in args.input_filenames:
            print( input_filename )
            convert_scores_format( input_filename, mode=args.mode )
