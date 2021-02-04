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
num_runs = 50
batch_size = 1024
epochs = 25
snapshot = False
warmup = 10
patience = 1

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--debug', action='store_true', help='Run in debug mode: Don\'t train any of the models and print '
                                                         'lots of diagnostic messages.')

args = parser.parse_args()

data_path = '/home/artemis/AugmExp/Data/type1/non_m4/train_only_lw_478k_clean.npy'

data = datasets.artemis_generator(data_path, batch_size=batch_size)

hp = {'optimizer': 'adam',
      'learning_rate': 0.001,
      'exp_decay': False,
      'amsgrad': True}

model_gen, hp = models.get_optimal_setup(hp)
hp['input_seq_length'] = 14

run_name = 'benchmark_foredeck/inp_14'
    
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
