import numpy as np
from tqdm import tqdm
from pathlib import Path
import tensorflow as tf
import warnings

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


def evaluate_snapshot_ensemble(family, x, y, result_dict=None):

    if not result_dict:
        results = {'smape': {}, 'mase*': {}}
    else:
        results = result_dict.copy()

    family = Path(family)
    num_trials = len(list(Path(family.parent).glob(family.name + '*')))
    
    family_preds = []

    for num in range(4):#range(num_trials):
        print(num)
        trial = str(family) + '__' + str(num)
        model_dir = trial + '/best_weights.h5'

        model = tf.keras.models.load_model(model_dir)

        preds = get_predictions(model, x)
        family_preds.append(preds)

        tf.keras.backend.clear_session()

        results['smape'][Path(trial).name] = np.nanmean(metrics.SMAPE(y, preds[:, -6:]))
        results['mase*'][Path(trial).name] = np.nanmean(metrics.MASE(x, y, preds[:, -6:]))

        ensemble_preds = np.median(np.array(family_preds), axis=0)

        results['smape'][family.name + '__ens__' + str(num + 1)] = np.nanmean(metrics.SMAPE(y, ensemble_preds[:, -6:]))
        results['mase*'][family.name + '__ens__' + str(num + 1)] = np.nanmean(metrics.MASE(x, y, ensemble_preds[:, -6:]))

    return results


def evaluate_snapshot_ensembles(families, x, y):
    results = {'smape': {}, 'mase*': {}}

    for family in tqdm(families):
        results = evaluate_snapshot_ensemble(family, x, y, results)

    return results


def find_untracked_trials(result_dir, tracked, verbose=False):

    all_trials = list(Path(result_dir).glob('*'))
    families, num_trials = np.unique(['__'.join(t.name.split('__')[:-1]) for t in all_trials], return_counts=True)
    untracked, undertracked = {}, {}

    for f, n in zip(families, num_trials):
        expected = tracked.get(f, 0)
        if not expected:
            untracked[f] = n
        elif expected < n:
            undertracked[f] = n
        elif expected > n:
            warnings.warn('More tracked trials recorded than found for family: {}\n'
                          '    Tracked: {}\n'
                          '    Found:   {}'.format(f, expected, n))

    tr = set(tracked.keys())
    ut = set(untracked.keys()).union(undertracked.keys())
    redundant = {k: tracked[k] for k in tr.difference(ut)}

    if verbose:
        l = max([len(f) for f in families])
        template = '        {:>' + str(l) + '} --> found: {}, expected: {}'
        print('Found {} unique trials belonging to {} families'.format(len(all_trials), len(families)))
        print('    Already tracked families:', len(tracked))
        print('    Untracked families found:', len(untracked))
        print('    Undertracked families:   ', len(undertracked))
        for f, n in undertracked.items():
            print(template.format(f, n, tracked[f]))
        print('    Redundant families:      ', len(undertracked))
        for f, n in redundant.items():
            print(template.format(f, n, tracked[f]))

    return untracked, undertracked, redundant

