import argparse
import tensorflow as tf

import datasets



def predict_N_ahead(model, batch, total_horizon=24, individual_horizon=6):

    results = []
    batch = batch.numpy()

    def predict_on_unscaled(batch):
        x = batch[..., 0]
        mn, mx = x.min(axis=1).reshape(-1, 1), x.max(axis=1).reshape(-1, 1)
        x_sc = (x - mn) / (mx - mn)
        pred = model(x_sc[..., np.newaxis])
        pred_us = pred[..., 0] * (mx - mn) + mn
        return pred_us[..., np.newaxis]

    for i in range(0, total_horizon, individual_horizon):
        preds = predict_on_unscaled(batch)
        batch = np.concatenate([batch[:, individual_horizon:, :], preds[:, :individual_horizon, :]], axis=1)
        results.append(preds[:, :individual_horizon])

    return np.concatenate(results, axis=1)


def make_augmented_dataset(model, data_it, total_horizon=24, individual_horizon=6, num_samples=235460,
                           batch_size=2048, real_insample=12):
    aug_batches = []
    for _ in tqdm(range(num_samples // batch_size + 1)):
        batch = data_it.__next__()
        full_pred = predict_N_ahead(model, batch[0], total_horizon=total_horizon,
                                    individual_horizon=individual_horizon)

        aug_batch = np.concatenate([batch[0][:, :real_insample, :],
                                    preds[:, real_insample:total_horizon, :]], axis=1)

        aug_batches.append(aug_batch)

    return np.concatenate(aug_batches)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-th', '--total-horizon', type=int, default=18, help='How long series we want.')
    parser.add_argument('-mh', '--model-horizon', type=int, default=18, help='How many predictions should the model '
                                                                             'predict at each step.')
    parser.add_argument('-n', '--num-samples', type=int, default=235460, help='How many samples to generate.')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode: Don\'t train any of the models and print lots of diagnostic messages.')

    args = parser.parse_args()

    data_path = 'data/yearly_{}.h5'.format(horizon)

    raise NotImplementedError('THIS SCRIPT IS IN PROGRESS. IT DOES NOT YET WORK PROPERLY.')
