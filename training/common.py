from pathlib import Path
import tensorflow as tf
import itertools
import utils
import os
import sys
from typing import Union

def train_model_snapshot(model: tf.keras.models.Model, train_set: tf.data.Dataset, run_name: str,
                         run_num: Union[str, int], cycles: int = 15, batch_size: int = 256) -> tf.keras.models.Model:
    """
    Train a model using Snapshot Ensembles with SGDA.

    :param model: An instance of a keras model
    :param train_set: A generator containing the training data
    :param run_name: Name of the current run
    :param run_num: Number of the current run (used to indicate multiple cold restarts)
    :param cycles: Number of cycles (i.e. how many times to store weights)
    :param batch_size: The batch size
    :return: The same instance of the model as the first argument
    """
    result_dir = 'results/'

    epochs = cycles + 5
    callbacks = [utils.callbacks.SnapshotWithAveraging(result_dir + str(run_name), n_cycles=cycles,
                                                       max_epochs=epochs, steps_to_average=100,
                                                       min_warmup_epochs=1, cold_start_id=run_num)]

    model.fit(train_set, epochs=epochs, steps_per_epoch=len(train_set)//batch_size+1, callbacks=callbacks)

    return model


def train_model_single(model: tf.keras.models.Model, train_set: tf.data.Dataset, run_name: str,
                       run_num: Union[str, int], epochs: int = 15, batch_size: int = 256, exp_decay: bool = True,
                       **kwargs) -> tf.keras.models.Model:
    """
    Trains a single keras model.

    :param model: An instance of a keras model
    :param train_set: A generator containing the training data
    :param run_name: Name of the current run
    :param run_num: Number of the current run (used to indicate multiple cold restarts)
    :param epochs: Number of epochs
    :param batch_size: The batch size
    :param exp_decay: Indication to use an exponential decay of the learning rate
    :param kwargs: Arguments to be passed to the checkpoint callback
           - warmup: how many epochs to wait before starting to save weights
           - patiance: every how many epochs to start storing weights
           - verbose: print every time weights are stored (doesn't work properly)
    :return: The same instance of the model as the first argument
    """

    steps_per_epoch = len(train_set) // batch_size + 1
    result_file = 'results/{}__{}/'.format(run_name, run_num) + 'weights_epoch_{epoch:03d}.h5'  # TODO: dynamic :03d

    callbacks = [utils.callbacks.SimpleModelCheckpoint(result_file, **kwargs)]

    if exp_decay:
        def scheduler(epoch, lr):
            return lr * tf.math.exp(-0.1)

        callbacks += [tf.keras.callbacks.LearningRateScheduler(schedule=scheduler)]

    model.fit(train_set, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)

    return model


def run_training(model_gen, hparams: dict, data: tf.data.Dataset, run_name: str, num_runs: int = 5, debug: bool = False,
                 snapshot: bool = False, **kwargs) -> None:
    """
    Function that handles the training of multiple models for use in ensembles.

    :param model_gen: Factory for producing model instances according to hparams
    :param hparams: A dictionary containing the hyperparameters used for creating the model instance
    :param data: A generator containing the training data
    :param run_name:  Number of the current run (used to indicate multiple cold restarts)
    :param num_runs: Number of epochs
    :param debug: Debug mode. Don't actually run training and store weights, but see if it works properly.
    :param snapshot: Indicate the use of snapshot ensembles or not.
    :param kwargs: Various training arguments, including:
           - exp_decay: Indication to use an exponential decay of the learning rate
           - epochs: Number of epochs for each training
           - batch_size: The batch size
           - warmup: how many epochs to wait before starting to save weights
           - patiance: every how many epochs to start storing weights
           - verbose: print every time weights are stored (doesn't work properly)
    """

    single_model_training_fcn = train_model_snapshot if snapshot else train_model_single

    if debug:
        model = model_gen(hparams)
        for x, y in data:
            print('Batch shapes:', x.shape, y.shape)
            model.train_on_batch(x, y)
            break
    else:
        for i in range(num_runs):
            print('Running experiment: "{}"\nIteration: {}/{}'.format(run_name.split('/')[-1], i+1, num_runs))
            model = model_gen(hparams)
            _ = single_model_training_fcn(model, data, run_name, run_num=i, **kwargs)
            del model
            tf.keras.backend.clear_session()


def make_runs(hparam_combinations_dict):
    """
    Function that generates all possible combinations from a dictionary.

    For example:

    >>> combs_dict = {'a': [1, 2, 3], 'b': [1, 2], 'c': [1, 2, 3, 4]}
    >>> combs = make_runs(combs_dict)

    >>> list(combs)

    >>> [{'a': 1, 'b': 1, 'c': 1},
    ...  {'a': 1, 'b': 1, 'c': 2},
    ...  {'a': 1, 'b': 1, 'c': 3},
    ...  {'a': 1, 'b': 1, 'c': 4},
    ...  {'a': 1, 'b': 2, 'c': 1},
    ...  {'a': 1, 'b': 2, 'c': 2},
    ...  {'a': 1, 'b': 2, 'c': 3},
    ...  {'a': 1, 'b': 2, 'c': 4},
    ...  {'a': 2, 'b': 1, 'c': 1},
    ...  {'a': 2, 'b': 1, 'c': 2},
    ...  {'a': 2, 'b': 1, 'c': 3},
    ...  {'a': 2, 'b': 1, 'c': 4},
    ...  {'a': 2, 'b': 2, 'c': 1},
    ...  {'a': 2, 'b': 2, 'c': 2},
    ...  {'a': 2, 'b': 2, 'c': 3},
    ...  {'a': 2, 'b': 2, 'c': 4},
    ...  {'a': 3, 'b': 1, 'c': 1},
    ...  {'a': 3, 'b': 1, 'c': 2},
    ...  {'a': 3, 'b': 1, 'c': 3},
    ...  {'a': 3, 'b': 1, 'c': 4},
    ...  {'a': 3, 'b': 2, 'c': 1},
    ...  {'a': 3, 'b': 2, 'c': 2},
    ...  {'a': 3, 'b': 2, 'c': 3},
    ...  {'a': 3, 'b': 2, 'c': 4}]

    :param hparam_combinations_dict: Dictionary with all values for each hyperparameter.
    :return: List with dictionary with all possible combinations.
    """

    names = hparam_combinations_dict.keys()
    combs = itertools.product(*hparam_combinations_dict.values())

    for c in combs:
        yield dict(zip(names, c))
