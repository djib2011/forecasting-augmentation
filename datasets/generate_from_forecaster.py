import numpy as np
import argparse
from tqdm import tqdm
import tensorflow as tf
from typing import Iterator
import os
import sys

sys.path.append(os.getcwd())

import datasets


def predict_N_ahead(model: tf.keras.models.Model, batch: tf.Tensor, total_pred_horizon: int = 24,
                    individual_horizon: int = 6) -> np.ndarray:
    """
    Function that chains a model's predictions in order to predict 'total_pred_horizon' steps ahead by predicting
    'individual_horizon' steps at a time.

    :param model: A tf.keras model instance (should be trained and have a horizon of at least individual_horizon)
    :param batch: A batch as a tensor
    :param total_pred_horizon: The total length of the predictions
    :param individual_horizon: How many time steps will each model predict.
    :return: The predictions. Will have a shape of (batch.shape[0], total_pred_horizon, 1).
    """

    results = [batch]
    batch = batch.numpy()

    def predict_on_unscaled(batch):
        x = batch[..., 0]
        mn, mx = x.min(axis=1).reshape(-1, 1), x.max(axis=1).reshape(-1, 1)
        x_sc = (x - mn) / (mx - mn)
        pred = model(x_sc[..., np.newaxis])
        pred_us = pred[..., 0] * (mx - mn) + mn
        return pred_us[..., np.newaxis]

    for i in range(0, total_pred_horizon, individual_horizon):
        preds = predict_on_unscaled(batch)
        batch = np.concatenate([batch[:, individual_horizon:, :], preds[:, :individual_horizon, :]], axis=1)

        results.append(preds[:, :individual_horizon])

    results = np.concatenate(results, axis=1)[..., 0]
    mn, mx = results.min(axis=1).reshape(-1, 1), results.max(axis=1).reshape(-1, 1)
    results = (results - mn) / (mx - mn)

    return results[..., np.newaxis]


def make_augmented_dataset(model: tf.keras.models.Model, data_it: Iterator, total_horizon: int = 24,
                           individual_horizon: int = 6, num_samples: int = 235460, batch_size: int = 2048,
                           real_insample: int = 12) -> np.ndarray:
    """
    This function will make use 'predict_N_ahead' to create the whole augmented dataset, in batches.

    Example:
        If the data_it produces batches of shape (batch_size, 18) and (batch_size, 6)
        and total_horizon==24, individual_horizon==3, real_insample==6 and num_samples==100000

        The augmented dataset will have a shape of (~100000, 24), where the first 6 data points are the last 6 of the
        insample and the remaining 18 are produced by chaining 3 predictions of the model at a time.

    :param model: A tf.keras model instance (should be trained and have a horizon of at least individual_horizon).
    :param data_it: An iterator for producing the batches.
    :param total_horizon: The total length of the predictions.
    :param individual_horizon: How many time steps will each model predict.
    :param num_samples: How many samples will should we augment approximately. Note that this number will be rounded up
                        to a factor of the batch size!
    :param batch_size: The batch size.
    :param real_insample: How many samples from the real series.
    :return:
    """

    aug_batches = []

    for _ in tqdm(range(num_samples // batch_size + 1)):
        batch = data_it.__next__()
        aug_batch = predict_N_ahead(model, batch[0], total_pred_horizon=total_horizon - real_insample,
                                    individual_horizon=individual_horizon)

        aug_batches.append(aug_batch)

    start = batch[0].shape[1] - real_insample

    return np.concatenate(aug_batches)[:, start:start + total_horizon]


if __name__ == '__main__':

    batch_size = 2048
    target_path = 'data/foreaug/'

    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--num-samples', type=int, default=235460, help='How many samples to generate.')
    parser.add_argument('-th', '--total-horizon', type=int, default=24, help='How long series we want.')
    parser.add_argument('-mh', '--model-horizon', type=int, default=6, help='How many predictions should the model '
                                                                            'predict at each step.')
    parser.add_argument('-r', '--real-insample', type=int, default=12, help='How many real datapoints we want in the '
                                                                            'beginning of the augmented series.')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode: Don\'t train any of the models and print lots of diagnostic messages.')

    args = parser.parse_args()

    data_path = 'data/yearly_24.h5'
    model_path = 'results/benchmark_all_windows/inp_18__41/weights_epoch_012.h5'

    data = datasets.seq2seq_generator(data_path, batch_size=batch_size).__iter__()
    model = tf.keras.models.load_model(model_path)

    aug_dataset = make_augmented_dataset(model, data, total_horizon=args.total_horizon, num_samples=args.num_samples,
                                         individual_horizon=args.model_horizon, real_insample=args.real_insample,
                                         batch_size=batch_size)

    name = 'foreaug__total_{}__real_{}__predby_{}.npy'.format(args.total_horizon, args.real_insample,
                                                              args.model_horizon)

    print('Augmented dataset shape:', aug_dataset.shape)

    np.save(target_path + name, aug_dataset)
