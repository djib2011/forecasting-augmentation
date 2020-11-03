import pickle as pkl
import numpy as np
import argparse

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
    n = 'data/yearly_{}_train_nw.pkl'.format(window)
else:
    n = 'data/yearly_{}_train.pkl'.format(window)
with open(n, 'rb') as f:
    data = pkl.load(f)

# Generate synthetic samples
s = [np.random.choice(np.arange(len(data[0])), args.num_samples, replace=True) for _ in range(args.combinations)]
syn = [np.sum([d[i] for i in s], axis=0) / args.combinations for d in data]

# Store new
if args.no_window:
    n = 'data/yearly_{}_train_aug_by_{}_num_{}_nw.pkl'.format(window, args.combinations, args.num_samples)
else:
    n = 'data/yearly_{}_train_aug_by_{}_num_{}.pkl'.format(window, args.combinations, args.num_samples)

with open(n, 'wb') as f:
    pkl.dump(syn, f)
