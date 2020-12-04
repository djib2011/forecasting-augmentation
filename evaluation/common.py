import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import tensorflow as tf
import warnings
import pickle as pkl

from utils import metrics
import datasets


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


def evaluate_family_with_multiple_weights(family, x, y, result_dict=None, desc=None, verbose=False):

    if not result_dict:
        results = {'smape': {}, 'mase*': {}}
    else:
        results = result_dict.copy()

    family = Path(family)
    trials = list(family.parent.glob(family.name + '*'))  # list for tqdm

    family_preds = None
    trial_ind = 0

    if verbose:
        print('Run name:', family.name)
        print('Family path:', str(family))
        print('Trials identified:', len(trials))
        for i, t in enumerate(trials):
            print('  {:>2d}. {}'.format(i, t))

    for trial in tqdm(trials, desc=desc):

        all_model_preds_in_trial = []

        trial = Path(trial)
        all_models_in_trial = list(trial.glob('*'))

        for epoch_ind, single_model in enumerate(all_models_in_trial):

            model = tf.keras.models.load_model(single_model)

            preds = get_predictions(model, x)

            if not family_preds:
                family_preds = [[] for _ in range(len(all_models_in_trial))]

            family_preds[epoch_ind].append(preds)
            all_model_preds_in_trial.append(preds)

            tf.keras.backend.clear_session()

            results['smape'][trial.name + '__epoch_{}'.format(epoch_ind + 1)] = np.nanmean(metrics.SMAPE(y, preds[:, -6:]))
            results['mase*'][trial.name + '__epoch_{}'.format(epoch_ind + 1)] = np.nanmean(metrics.MASE(x, y, preds[:, -6:]))

            ensemble_preds = np.median(np.array(all_model_preds_in_trial), axis=0)

            results['smape']['ens__' + trial.name + '__epoch_{}'.format(epoch_ind + 1)] = np.nanmean(metrics.SMAPE(y, ensemble_preds[:, -6:]))
            results['mase*']['ens__' + trial.name + '__epoch_{}'.format(epoch_ind + 1)] = np.nanmean(metrics.MASE(x, y, ensemble_preds[:, -6:]))

        trial_ind += 1

        for i, epoch_preds in enumerate(family_preds):

            preds = np.median(np.array(epoch_preds), axis=0)

            results['smape']['ens__' + family.name + '____epoch_{}'.format(i)] = np.nanmean(metrics.SMAPE(y, preds[:, -6:]))
            results['mase*']['ens__' + family.name + '____epoch_{}'.format(i)] = np.nanmean(metrics.MASE(x, y, preds[:, -6:]))

    return results


def evaluate_snapshot_ensemble(family, x, y, result_dict=None, desc=None):

    if not result_dict:
        results = {'smape': {}, 'mase*': {}}
    else:
        results = result_dict.copy()

    family = Path(family)
    trials = list(family.parent.glob(family.name + '*'))  # list for tqdm

    family_preds = []
    num = 0

    for trial in tqdm(trials, desc=desc):

        num += 1
        model_dir = trial / 'best_weights.h5'

        model = tf.keras.models.load_model(model_dir)

        preds = get_predictions(model, x)
        family_preds.append(preds)

        tf.keras.backend.clear_session()

        results['smape'][family.name + '__' + str(num)] = np.nanmean(metrics.SMAPE(y, preds[:, -6:]))
        results['mase*'][family.name + '__' + str(num)] = np.nanmean(metrics.MASE(x, y, preds[:, -6:]))

        ensemble_preds = np.median(np.array(family_preds), axis=0)

        results['smape'][family.name + '__ens__' + str(num)] = np.nanmean(metrics.SMAPE(y, ensemble_preds[:, -6:]))
        results['mase*'][family.name + '__ens__' + str(num)] = np.nanmean(metrics.MASE(x, y, ensemble_preds[:, -6:]))

    return results


def evaluate_multiple_families(families, x, y, snapshot=False):
    results = {'smape': {}, 'mase*': {}}

    family_eval_fcn = evaluate_snapshot_ensemble if snapshot else evaluate_family_with_multiple_weights

    if len(families) > 1 and not isinstance(families, str):
        num_digits = str(len(str(len(families))))
        template = 'family {:>' + num_digits + '} of {:<' + num_digits + '}'
    else:
        template = ''
        families = [families]

    for i, family in enumerate(families):

        results = family_eval_fcn(family, x, y, results, desc=template.format(i+1, len(families)))

        with open('/tmp/{}.pkl'.format(Path(family).name), 'wb') as f:
            pkl.dump(results, f)

    return results


def find_untracked_trials(result_dir, tracked=None, exclude_pattern=None, verbose=False):

    if not tracked:
        tracked = {}

    all_trials = list(Path(result_dir).glob('*'))
    if exclude_pattern:
        all_trials = [p for p in all_trials if exclude_pattern not in p.name]

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


def create_results_df_multi_weights(results, columns):

    keys_original = [k for k in results['smape'].keys()]
    keys = [k.replace('ens__', '') for k in keys_original]
    df = pd.DataFrame([k.split('__') for k in keys], columns=columns + ['num', 'epoch'])

    df['ensemble'] = ['ens__' in k for k in keys_original]
    df['smape'] = [results['smape'][k] if results['smape'][k] else np.nan for k in keys_original]
    df['mase*'] = [results['mase*'][k] if results['mase*'][k] else np.nan for k in keys_original]

    for column in columns + ['epoch']:
        try:
            df[column] = df[column].apply(lambda x: x.split('_')[1])
        except IndexError:
            raise IndexError('Trying to split column {}'.format(column))

    return df


def create_results_df_snapshot(results, columns):

    single_keys = [k for k in results['smape'].keys() if 'ens' not in k]
    df1 = pd.DataFrame([k.split('__') for k in single_keys], columns=columns + ['num'])
    df1['ensemble'] = False
    df1['smape'] = [results['smape'][k] if results['smape'][k] else np.nan for k in single_keys]
    df1['mase*'] = [results['mase*'][k] if results['mase*'][k] else np.nan for k in single_keys]

    ens_keys = [k for k in results['smape'].keys() if 'ens' in k]
    df2 = pd.DataFrame([k.replace('__ens', '').split('__') for k in ens_keys], columns=columns + ['num'])
    df2['ensemble'] = True
    df2['smape'] = [results['smape'][k] if results['smape'][k] else np.nan for k in ens_keys]
    df2['mase*'] = [results['mase*'][k] if results['mase*'][k] else np.nan for k in ens_keys]

    df = pd.concat([df1, df2])

    for column in columns:
        try:
            df[column] = df[column].apply(lambda x: x.split('_')[1])
        except IndexError:
            raise IndexError('Trying to split column {}'.format(column))

    return df


def run_evaluation(result_dir, report_dir, columns, exclude_pattern=None, return_results=False,
                   snapshot=False, debug=False):

    tracked_file = (Path(report_dir) / 'tracked.pkl')
    if tracked_file.exists():
        with open(tracked_file, 'rb') as f:
            tracked = pkl.load(f)
    else:
        tracked = {}

    untracked, undertracked, _ = find_untracked_trials(result_dir, tracked, exclude_pattern=exclude_pattern,
                                                       verbose=True)
    untracked.update(undertracked)
    tracked.update(untracked)
    
    X_test, y_test = datasets.load_test_set()

    families = [Path(result_dir) / u for u in untracked]

    if debug:
        if untracked:
            print('Untracked trials:')
            for i, t in enumerate(untracked):
                print('{:>2}. {}'.format(i+1, t))

        if undertracked:
            print('Undertracked trials:')
            for i, t in enumerate(undertracked):
                print('{:>2}. {}'.format(i+1, t))

    else:
        results = evaluate_multiple_families(families, X_test, y_test, snapshot=snapshot)
        
        if return_results:
            return results, tracked

        results_df_fcn = create_results_df_snapshot if snapshot else create_results_df_multi_weights

        result_df_file = (Path(report_dir) / 'results.csv')
        if result_df_file.exists():
            df = pd.read_csv(result_df_file)
            df = pd.concat([df, results_df_fcn(results, columns=columns)])
        else:
            df = results_df_fcn(results, columns=columns)
        
        # TODO: make dir if it doesn't exist
        df.to_csv(result_df_file, index=False)

        with open(tracked_file, 'wb') as f:
            pkl.dump(tracked, f)
