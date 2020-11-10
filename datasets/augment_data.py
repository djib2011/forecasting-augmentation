import pickle as pkl
import numpy as np
import argparse
import os
import sys

if os.getcwd().endswith('datasets'):
    os.chdir('..')

sys.path.append(os.getcwd())

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_len', type=int, default=12, help='Insample length.')
parser.add_argument('-n', '--num_samples', type=int, default=100000, help='Number of samples after augmentation.')
parser.add_argument('-c', '--combinations', type=int, default=2, help='Number of series to combine.')
parser.add_argument('--no_window', action='store_true', help='Use the no-window version of the data')

args = parser.parse_args()

window = args.input_len + 6

# Load data
if args.no_window:
    n = 'data/yearly_{}_X_nw.csv'.format(window)
else:
    n = 'data/yearly_{}_X.csv'.format(window)

with open(n, 'rb') as f:
    data = np.loadtxt(n, delimiter=',')

# Generate synthetic samples
s = [np.random.choice(np.arange(len(data[0])), args.num_samples, replace=True) for _ in range(args.combinations)]
syn = np.array([np.sum([d[i] for i in s], axis=0) / args.combinations for d in data])

# Store new
if args.no_window:
    n = 'data/yearly_{}_aug_by_{}_num_{}_nw.csv'.format(window, args.combinations, args.num_samples)
else:
    n = 'data/yearly_{}_aug_by_{}_num_{}.csv'.format(window, args.combinations, args.num_samples)

print('Saving data at:', n)
np.savetxt(n, syn, delimiter=',')
