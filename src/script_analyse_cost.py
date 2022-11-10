import csv
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    args = parser.parse_args()

    with open(args.filename, 'r') as f:
        # import csv file
        data = csv.reader(f, delimiter=',')

        headings = data.__next__()
        cost_index = headings.index('cost')
        cost_vases_contact = headings.index('cost_vases_contact')
        cost_hazards_constact = headings.index('cost_hazards')
        constraint_index = headings.index('constraint')
        time_index = headings.index('time')

        for row in data:
            c_vases    = row[cost_vases_contact]
            c_hazards  = row[cost_hazards_constact]
            constraint = row[constraint_index]
            time = row[time_index]
            if c_vases != '0.0' or c_hazards != '0.0' or constraint != '1.0':
                print( time, c_vases, c_hazards, constraint )
            
