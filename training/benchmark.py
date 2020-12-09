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
parser.add_argument('--debug', action='store_true', help='Run in debug mode: Don\'t train any of the models and print '
                                                         'lots of diagnostic messages.')

args = parser.parse_args()

data_path = 'data/yearly_{}_nw.h5'.format(args.input_len + 6)
run_name = 'benchmark2/inp_{}_nw'.format(args.input_len)

data = datasets.seq2seq_generator(data_path, batch_size=batch_size)

hp = {'base_layer_size': 128,
      'direction': 'bi',
      'depth': 2,
      'input_seq_length': 18,
      'output_seq_length': 6}

model_gen = models.get(family='sequential', type=hp['direction'], depth=hp['depth'])

training.run_training(model_gen, hp, data, run_name, num_runs=10, debug=args.debug, epochs=20, snapshot=False)



data_path = 'data/yearly_{}.h5'.format(args.input_len + 6)
run_name = 'benchmark2/inp_{}'.format(args.input_len)

data = datasets.seq2seq_generator(data_path, batch_size=batch_size)

hp = {'base_layer_size': 128,
      'direction': 'bi',
      'depth': 2,
      'input_seq_length': 18,
      'output_seq_length': 6}

model_gen = models.get(family='sequential', type=hp['direction'], depth=hp['depth'])

training.run_training(model_gen, hp, data, run_name, num_runs=10, debug=args.debug, epochs=20, snapshot=False)
