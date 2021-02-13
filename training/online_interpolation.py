import os
import sys

if os.getcwd().endswith('training'):
    os.chdir('..')
sys.path.append(os.getcwd())

import tensorflow as tf
import datasets
import models
import os
import argparse

import training


# Global configs
input_len = 18
num_runs = 50
batch_size = 2048
epochs = 25
snapshot = False
warmup = 10
patience = 1

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--debug', action='store_true', help='Run in debug mode: Don\'t train any of the models and print '
                                                         'lots of diagnostic messages.')

args = parser.parse_args()

data_path = '/home/artemis/AugmExp/Data/type1/train_in18_win1000.npy'
run_name = 'online_interpolation/dset_235k__aug_0.5'

data = datasets.artemis_generator(data_path, batch_size=batch_size)

model_gen, hp = models.get_optimal_setup()

training.run_training(model_gen, hp, data, run_name, num_runs=num_runs, debug=args.debug, batch_size=batch_size,
                      epochs=epochs, snapshot=snapshot, warmup=warmup, patience=patience)

