import os
import sys

if os.getcwd().endswith('training'):
    os.chdir('..')
sys.path.append(os.getcwd())

import datasets
import models
import argparse

import training

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_len', type=int, default=18, help='Insample length.')
parser.add_argument('--debug', action='store_true', help='Run in debug mode: Don\'t train any of the models and print '
                                                         'lots of diagnostic messages.')

args = parser.parse_args()

data_path = 'data/yearly_{}.h5'.format(args.input_len + 6)

data = datasets.seq2seq_generator(data_path, batch_size=1024)


hp = {'base_layer_size': 32,
      'direction': 'bi',
      'depth': 3,
      'input_seq_length': 18,
      'output_seq_length': 6}

model_gen = models.get(family='sequential', type=hp['direction'], depth=hp['depth'])

# First run with snapshot ensemble
model = model_gen(hp)
run_name = 'test_snapshot/with_snapshot'
training.run_training(model, data, run_name, num_runs=5, debug=args.debug, snapshot=True)

# Run without snapshot ensemble
del model
model = model_gen(hp)
run_name = 'test_snapshot/no_snapshot'
training.run_training(model, data, run_name, num_runs=10, debug=args.debug, epochs=50, snapshot=False)
