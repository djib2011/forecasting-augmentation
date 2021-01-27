"""
Script that creates datasets of generated timeseries using various combination strategies.
"""

import numpy as np
import os
import sys
from tqdm import tqdm
from pathlib import Path

if os.getcwd().endswith('datasets'):
    os.chdir('..')

sys.path.append(os.getcwd())

import datasets

original_data_path = '/home/artemis/AugmExp/Data/train_in18_all_windows.npy'
target_data_dir = 'data/combinations/'

# Load data
data = np.load(original_data_path)[:, -24:]

combinations = [2, 3, 5]
num_samples = 235460
num_runs = 20

for i in tqdm(range(1, num_runs+1)):

    for c in combinations:

        aug_data = datasets.make_combinations(data, num_samples=num_samples, num_combs=c, seed=i)
        aug_data = datasets.normalize_data(aug_data)

        file_name = Path(target_data_dir) / 'comb_{}'.format(c) / 'aug_data_{}.npy'.format(i)

        if not file_name.parent.is_dir():
            os.makedirs(str(file_name.parent))

        np.save(str(file_name), aug_data)
