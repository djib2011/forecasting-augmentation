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
num_runs = 4
batch_size = 2048
epochs = 15
snapshot = False
warmup = 0
patience = 1

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_len', type=int, default=18, help='Insample length.')
parser.add_argument('-n', '--num_samples', type=str, default='all_windows', help='Number of samples in the training set.')
parser.add_argument('--debug', action='store_true', help='Run in debug mode: Don\'t train any of the models and print '
                                                         'lots of diagnostic messages.')

args = parser.parse_args()

data_path = '/home/artemis/AugmExp/Data/train_in18_{}.npy'.format(args.num_samples)
run_name = 'windows/inp_{}__numsamples_{}'.format(args.input_len, args.num_samples)

data = datasets.artemis_generator(data_path, batch_size=batch_size)

hp = {'base_layer_size': 256, 'input_seq_length': args.input_len, 'output_seq_length': 6}
model_gen = models.sequential.bidirectional_2_layer

# model_gen, hparams = models.get_optimal_setup()

training.run_training(model_gen, hp, data, run_name, num_runs=num_runs, debug=args.debug, batch_size=batch_size,
                      epochs=epochs, snapshot=snapshot, warmup=warmup, patience=patience)
