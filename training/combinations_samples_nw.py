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

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_len', type=int, default=18, help='Insample length.')
parser.add_argument('-c', '--combinations', type=int, default=2,
                    help='Number of series combined to produce augmentations.')
parser.add_argument('-n', '--num_samples', type=int, default=23000, help='Number of augmented samples.')
parser.add_argument('--debug', action='store_true', help='Run in debug mode: Don\'t train any of the models and print '
                                                         'lots of diagnostic messages.')

args = parser.parse_args()

data_path = 'data/yearly_{}_aug_by_{}_num_{}_nw.h5'.format(args.input_len + 6, args.combinations, args.num_samples)

data = datasets.seq2seq_generator(data_path, batch_size=1024)

run_name = 'comb_nw/inp_{}__num_{}__comb_{}'.format(args.input_len, args.num_samples, args.combinations)

hp = {'base_layer_size': 16, 'input_seq_length': args.input_len, 'output_seq_length': 6}
model = models.sequential.bidirectional_ae_2_layer(hp)

for i in range(10):
    if args.debug:
        for x, y in data:
            print('Batch shapes:', x.shape, y.shape)
            model.train_on_batch(x, y)
            break
    else:
        _ = training.train_model(model, data, run_name, run_num=i)
