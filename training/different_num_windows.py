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


batch_size = 2048
epochs = 15

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_len', type=int, default=18, help='Insample length.')
parser.add_argument('-n', '--num_samples', type=int, default=2, help='Number of samples in the training set.')
parser.add_argument('--debug', action='store_true', help='Run in debug mode: Don\'t train any of the models and print '
                                                         'lots of diagnostic messages.')

args = parser.parse_args()

data_path = '~artemis/AugExp/Data/yyyy_{}_xxxx.npy'.format(args.num_samples)
run_name = 'windows/inp_{}__numsamples_{}'.format(args.input_len, args.num_samples)

data = datasets.artemis_generator(data_path, batch_size=batch_size)

# hp = {'base_layer_size': 16, 'input_seq_length': args.input_len, 'output_seq_length': 6}
# model = models.sequential.bidirectional_3_layer(hp)

# model_gen, hparams = models.get_optimal_setup()

training.run_training(model_gen, hparams, data, run_name, num_runs=10, debug=args.debug, snapshot=False)