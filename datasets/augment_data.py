import pickle as pkl
import numpy as np
import argparse
import os
import sys
import h5py

if os.getcwd().endswith('datasets'):
    os.chdir('..')

sys.path.append(os.getcwd())

import datasets

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_len', type=int, default=18, help='Insample length.')
parser.add_argument('-n', '--num_samples', type=int, default=14000, help='Number of samples after augmentation.')
parser.add_argument('-c', '--combinations', type=int, default=2, help='Number of series to combine.')
parser.add_argument('--no_window', action='store_true', help='Use the no-window version of the data')

args = parser.parse_args()

window = args.input_len + 6

# Load data
if args.no_window:
    n = 'data/yearly_{}_nw.h5'.format(window)
else:
    n = 'data/yearly_{}.h5'.format(window)

with h5py.File(n, 'r') as hf:
    X = np.array(hf.get('X'))
    y = np.array(hf.get('y'))

data = np.c_[X, y]
del X, y

# Generate synthetic samples
syn = datasets.make_combinations(data, num_samples=args.num_samples, num_combs=args.combinations, )
syn = datasets.normalize_data(syn)

X = syn[:, :-6]
y = syn[:, -6:]
del syn

# Locations to store augmented
if args.no_window:
    n = 'data/aug_nw/yearly_{}_aug_by_{}_num_{}_nw.h5'.format(window, args.combinations, args.num_samples)
else:
    n = 'data/aug_nw/yearly_{}_aug_by_{}_num_{}.h5'.format(window, args.combinations, args.num_samples)

print('Saving files to:', n)

with h5py.File(n, 'w') as hf:
    hf.create_dataset('X', data=X)
    hf.create_dataset('y', data=y)
