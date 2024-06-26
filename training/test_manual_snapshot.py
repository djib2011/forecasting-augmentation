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
num_runs = 1
batch_size = 2048
epochs = 201
snapshot = False
warmup = 0
patience = 1

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_len', type=int, default=18, help='Insample length.')
parser.add_argument('--debug', action='store_true', help='Run in debug mode: Don\'t train any of the models and print '
                                                         'lots of diagnostic messages.')

args = parser.parse_args()

data_path = 'data/yearly_{}.h5'.format(args.input_len + 6)

data = datasets.seq2seq_generator(data_path, batch_size=1024)

hparams = {'base_layer_size': 256,
           'direction': 'bi',
           'depth': 2,
           'input_seq_length': 18,
           'output_seq_length': 6}

model_gen = models.get(family='sequential', type=hparams['direction'], depth=hparams['depth'])

run_name = 'test_manual_snapshot/inp_{}__dir_{}__size_{}__depth_{}'.format(hparams['input_seq_length'],
                                                                           hparams['direction'],
                                                                           hparams['base_layer_size'],
                                                                           hparams['depth'])

training.run_training(model_gen, hparams, data, run_name, num_runs=num_runs, debug=args.debug, batch_size=batch_size,
                      epochs=epochs, snapshot=snapshot, warmup=warmup, patience=patience)
