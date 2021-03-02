import argparse
import tensorflow as tf

import datasets


def predict_N_ahead(model, batch, total_horizon=24, individual_horizon=6):

    result = []

    for i in range(0, total_horizon, individual_horizon):
        preds = model.predict(batch)
        batch = np.concatenate(batch[:, batch.shape[1]-individual_horizon:, :], preds[:, :individual_horizon, :], axis=1)
        result.append(preds[:individual_horizon])

    return np.concatenate(results, axis=1)


def make_augmented_dataset(model_loc, original_dataset_loc, total_horizon=24, individual_horizon=6, num_samples=235460,
                           batch_size=2048):

    data = datasets.seq2seq_generator(original_dataset_loc, batch_size=batch_size)

    model = tf.keras.models.load_model(model_loc)

    aug_batches = []
    for _ in tqdm(range(num_samples//batch_size+1)):
        batch = data.__next__()
        aug_batches.append(predict_N_ahead(model, batch, total_horizon=total_horizon,
                                           individual_horizon=individual_horizon))

    return np.concatenate(aug_batches, axis=1)


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



