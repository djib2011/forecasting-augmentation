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
num_runs=10
batch_size=2048
epochs=10
snapshot=False

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_len', type=int, default=18, help='Insample length.')
parser.add_argument('--debug', action='store_true', help='Run in debug mode: Don\'t train any of the models and print '
                                                         'lots of diagnostic messages.')

args = parser.parse_args()

data_path = 'data/yearly_{}.h5'.format(args.input_len + 6)

data = datasets.seq2seq_generator(data_path, batch_size=1024)

hp_comb_dict = {'base_layer_size': [16, 32, 64, 128],
                'direction': ['uni', 'bi'],
                'depth': [2, 3, 4],
                'input_seq_length': [18],
                'output_seq_length': [6]}

hp_generator = training.make_runs(hp_comb_dict)

for hp in hp_generator:

    model_gen = models.get(family='sequential', type=hp['direction'], depth=hp['depth'])
    model = model_gen(hp)
    run_name = 'lstm_tuning_windows/inp_{}__dir_{}__size_{}__depth_{}'.format(hp['input_seq_length'],
                                                                              hp['direction'],
                                                                              hp['base_layer_size'],
                                                                              hp['depth'])
    if args.debug:
        print('run name:', run_name)

    training.run_training(model, data, run_name, num_runs=num_runs, debug=args.debug,
                          batch_size=batch_size, epochs=epochs, snapshot=snapshot)
