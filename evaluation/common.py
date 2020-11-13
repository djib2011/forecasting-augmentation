import numpy as np
from tqdm import tqdm
from pathlib import Path
import tensorflow as tf
from utils import metrics


def get_predictions(model, X, batch_size=256):
    preds = []

    def predict_on_unscaled(x):
        mn, mx = x.min(axis=1).reshape(-1, 1), x.max(axis=1).reshape(-1, 1)
        x_sc = (x - mn) / (mx - mn)
        pred = model(x_sc[..., np.newaxis])
        return pred[..., 0] * (mx - mn) + mn

    for i in range(len(X) // batch_size):
        x = X[i * batch_size:(i + 1) * batch_size]
        preds.append(predict_on_unscaled(x))

    x = X[(i + 1) * batch_size:]
    preds.append(predict_on_unscaled(x))

    return np.vstack(preds)


def evaluate_snapshot_ensemble(family, x, y, results=None):

    if not results:
        results = {'smape': {}, 'mase*': {}}

    num_trials = len(list(Path(family).glob('*')))

    family_preds = []

    for num in range(num_trials):

        trial = str(family) + '__' + str(num)
        model_dir = trial + '/best_weights.h5'


        model = tf.keras.models.load_model(model_dir)

        preds = get_predictions(model, x)
        family_preds.append(preds)

        tf.keras.backend.clear_session()

        results['smape'][Path(trial).name] = np.nanmean(metrics.SMAPE(y, preds[:, -6:]))
        results['mase*'][Path(trial).name] = np.nanmean(metrics.MASE(x, y, preds[:, -6:]))

        ensemble_preds = np.median(np.array(family_preds), axis=0)

        results['smape'][family.name + '_ens__' + str(num + 1)] = np.nanmean(metrics.SMAPE(y, ensemble_preds[:, -6:]))
        results['mase*'][family.name + '_ens__' + str(num + 1)] = np.nanmean(metrics.MASE(x, y, ensemble_preds[:, -6:]))

    return results


def evaluate_snapshot_ensembles(families, x, y):
    results = {'smape': {}, 'mase*': {}}

    for family in tqdm(families):
        results = evaluate_snapshot_ensemble(family, results, x, y)
