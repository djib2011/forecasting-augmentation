from pathlib import Path
import tensorflow as tf
import itertools
import utils
import os
import sys


def train_model_snapshot(model, train_set, run_name, run_num, cycles=15, batch_size=256):
    result_dir = 'results/'

    epochs = cycles + 5
    callbacks = [utils.callbacks.SnapshotWithAveraging(result_dir + str(run_name), n_cycles=cycles,
                                                       max_epochs=epochs, steps_to_average=100,
                                                       min_warmup_epochs=1, cold_start_id=run_num)]

    model.fit(train_set, epochs=epochs, steps_per_epoch=len(train_set)//batch_size+1, callbacks=callbacks)

    return model


def train_model_single(model, train_set, run_name, run_num, epochs=15, batch_size=256, **kwargs):

    steps_per_epoch = len(train_set)//batch_size+1
    result_file = 'results/{}__{}/'.format(run_name, run_num) + 'weights_epoch_{epoch:03d}.h5'  # TODO: dynamic :03d

    callbacks = [utils.callbacks.SimpleModelCheckpoint(result_file, **kwargs)]

    model.fit(train_set, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)

    return model


def run_training(model_gen, hparams, data, run_name, num_runs=5, debug=False, snapshot=False, **kwargs):

    single_model_training_fcn = train_model_snapshot if snapshot else train_model_single

    if debug:
        model = model_gen(hparams)
        for x, y in data:
            print('Batch shapes:', x.shape, y.shape)
            model.train_on_batch(x, y)
            break
    else:
        model = model_gen(hparams)
        for i in range(num_runs):
            # model = model_gen(hparams)
            _ = single_model_training_fcn(model, data, run_name, run_num=i, **kwargs)
            # del model
            # tf.keras.backend.clear_session()


def make_runs(hparam_combinations_dict):

    names = hparam_combinations_dict.keys()
    combs = itertools.product(*hparam_combinations_dict.values())

    for c in combs:
        yield dict(zip(names, c))

