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
import utils

# Global configs
num_runs = 4
batch_size = 2048
epochs = 15
snapshot = False
warmup = 0
patience = 1

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--debug', action='store_true', help='Run in debug mode: Don\'t train any of the models and print '
                                                         'lots of diagnostic messages.')

args = parser.parse_args()

data_path = 'data/yearly_{}.h5'.format(18 + 6)

conb_dict = {'aug_perc': [0.25, 0.5, 0.75],
             'input_seq_length': [18],
             'output_seq_length': [6]}

hp_generator = training.make_runs(comb_dict)

for hp in hp_generator:

    model_gen, hp = models.get_optimal_setup(hp)

    data = datasets.seq2seq_generator(data_path, batch_size=batch_size, augmentation=hp['aug_perc'])

    run_name = 'online_augmentation/dset_{}__perc_{}'.format('235k', str(hp['aug_perc']))

    if args.debug:
        print('run name:', run_name)
        model = model_gen(hp)
        for x, y in data:
            model.train_on_batch(x, y)
            break
            
    else:
        for i in range(num_runs):
            model = model_gen(hp)

            model = training.train_model_single(model, data, run_name, run_num=i, exp_decay=False,
                                                batch_size=batch_size, epochs=epochs, snapshot=snapshot, warmup=warmup,
                                                patience=patience)

            del model
            tf.keras.backend.clear_session()
