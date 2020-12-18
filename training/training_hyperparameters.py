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

data = datasets.seq2seq_generator(data_path, batch_size=batch_size)

adam_comb_dict = {'optimizer': ['adam'],
                  'learning_rate': [0.01, 0.02, 0.05],
                  'exp_decay': [True],
                  'amsgrad': [False],
                  'direction': ['bi'],
                  'base_layer_size': [128],
                  'depth': [2],
                  'input_seq_length': [18],
                  'output_seq_length': [6]}

hp_generator = training.make_runs(adam_comb_dict)

for hp in hp_generator:

    model_gen = models.get(family='sequential', type=hp['direction'], depth=hp['depth'])

    run_name = 'training_hyperparameters/opt_{}__lr_{}__ams_{}__decay_{}'.format(hp['optimizer'],
                                                                                 hp['learning_rate'],
                                                                                 hp['amsgrad'],
                                                                                 hp['exp_decay'])

    if args.debug:
        print('run name:', run_name)
        model = model_gen(hp)
        opt = utils.optimizers.build_optimizer(hp['optimizer'], learning_rate=hp['learning_rate'],
                                               amsgrad=hp['amsgrad'])
        model.compile(loss='mae', optimizer=opt, metrics=['mae', 'mse'])
        for x, y in data:
            model.train_on_batch(x, y)
            break
            
    else:
        for i in range(num_runs):
            model = model_gen(hp)
            opt = utils.optimizers.build_optimizer(hp['optimizer'], learning_rate=hp['learning_rate'],
                                                   amsgrad=hp['amsgrad'])

            model.compile(loss='mae', optimizer=opt, metrics=['mae', 'mse'])

            model = training.train_model_single(model, data, run_name, run_num=i, exp_decay=hp['exp_decay'],
                                                batch_size=batch_size, epochs=epochs, snapshot=snapshot, warmup=warmup,
                                                patience=patience)
            del model
            tf.keras.backend.clear_session()
