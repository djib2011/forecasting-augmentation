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
batch_size = 1024
epochs = 18
snapshot = False
warmup = 3
patience = 1

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_len', type=int, default=18, help='Insample length.')
parser.add_argument('--debug', action='store_true', help='Run in debug mode: Don\'t train any of the models and print '
                                                         'lots of diagnostic messages.')

args = parser.parse_args()

data_path = 'data/yearly_{}.h5'.format(args.input_len + 6)

data = datasets.seq2seq_generator(data_path, batch_size=batch_size)

adam_comb_dict = {'optimizer': ['adam'],
                  'learning_rate': [0.0005, 0.001, 0.005],
                  'exp_decay': [False, True],
                  'amsgrad': [False, True]}

hp_generator = training.make_runs(adam_comb_dict)


for hp in hp_generator:

    model_gen, hp = models.get_optimal_setup(hp)

    run_name = 'fine_tune_best/opt_{}__lr_{}__ams_{}__decay_{}'.format(hp['optimizer'],
                                                                       hp['learning_rate'],
                                                                       hp['amsgrad'],
                                                                       hp['exp_decay'])

    if args.debug:
        print('run name:', run_name)

    training.run_training(model_gen, hp, data, run_name, num_runs=num_runs, debug=args.debug, batch_size=batch_size,
                          epochs=epochs, snapshot=snapshot, warmup=warmup, patience=patience)
