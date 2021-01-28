import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import tensorflow as tf
import warnings
import pickle as pkl
from typing import Union, Sequence

from utils import metrics, deprecated
import datasets


def get_predictions(model: tf.keras.models.Model, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """
    Compute a model's predictions on a given dataset of timeseries.

    Note: model is assumed to be trained on normalized data.

    :param model: A tf.keras model.
    :param X: A dataset to generate predictions on
    :param batch_size: The batch size.
    :return: An array of the model's predictions.
    """
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


def evaluate_family_with_multiple_weights(family: Union[str, Path], x: np.ndarray, y: np.ndarray,
                                          result_dict: dict = None, desc: str = None, verbose: bool = False,
                                          batch_size: int = 1024) -> dict:
    """
    Run full evaluation on a family of models, each having multiple weights (i.e. one for each epoch).

    The function will:

        1) Find all models in the family and all instances of the specific model.
        2) Get the predictiond for each instance.
        3) Compute the sMAPE and MASE estimate of each instance.
        4) Compute the ensemble predictions across instances of a specific model, as well as their sMAPE and MASE*.
        5) Compute the ensemble predictions across different models, per epoch, as well as their sMAPE and MASE*.
        6) Compute the ensemble predictions of the last epoch ensembles of (4), as well as their sMAPE and MASE*.


    The function assumes that the models are saved as:

        results_dir
            |
            |________experiment1__0
            |              |
            |              |________weights_epoch_0.h5
            |              |________weights_epoch_1.h5
            |              |________weights_epoch_2.h5
            |                      ...
            |________experiment1__1
            |               |
            |               |________weights_epoch_0.h5
            |               |________weights_epoch_1.h5
            |               |________weights_epoch_2.h5
            |                        ...
                        ...


    :param family: Location and name of the family of models. In the example structure above, the family would be:
                   'results_dir/experiment'

    :param x: Numpy array containing insample data
    :param y: Numpy array containing out-of-sample data
    :param result_dict: Dictionary to store the reuslts in (optional)
    :param desc: Description for the tqdm (usually the experiment's name)
    :param verbose: Option to print messages
    :param batch_size: The batch size
    :return: A dictionary containing the sMAPE and MASE* of the individual models and their ensembles.
    """

    if not result_dict:
        results = {'smape': {}, 'mase*': {}}
    else:
        results = result_dict.copy()

    family = Path(family)
    trials = sorted(family.parent.glob(family.name + '*'), key=lambda x: int(x.name.split('__')[-1]))

    if verbose:
        print('Run name:', family.name)
        print('Family path:', str(family))
        print('Trials identified:', len(trials))
        for i, t in enumerate(trials):
            print('  {:>2d}. {}'.format(i, t))

    family_preds = None
    ensemble_preds_all_trials = []

    for trial in tqdm(trials, desc=desc):

        all_models_in_trial = sorted(trial.glob('*'), key=lambda x: x.name.split('/')[-1])

        if not family_preds:
            family_preds = [[] for _ in range(len(all_models_in_trial))]

        all_model_preds_in_trial = []

        for epoch_ind, single_model in enumerate(all_models_in_trial):

            model = tf.keras.models.load_model(single_model)

            preds = get_predictions(model, x, batch_size=batch_size)

            family_preds[epoch_ind].append(preds)

            results['smape'][trial.name + '__epoch_{}'.format(epoch_ind + 1)] = np.nanmean(
                metrics.SMAPE(y, preds[:, -6:]))
            results['mase*'][trial.name + '__epoch_{}'.format(epoch_ind + 1)] = np.nanmean(
                metrics.MASE(x, y, preds[:, -6:]))

            all_model_preds_in_trial.append(preds)

            ensemble_preds = np.median(np.array(all_model_preds_in_trial), axis=0)

            results['smape']['ens__' + trial.name + '__epoch_{}'.format(epoch_ind + 1)] = np.nanmean(
                metrics.SMAPE(y, ensemble_preds[:, -6:]))
            results['mase*']['ens__' + trial.name + '__epoch_{}'.format(epoch_ind + 1)] = np.nanmean(
                metrics.MASE(x, y, ensemble_preds[:, -6:]))

            del model
            tf.keras.backend.clear_session()

        ensemble_preds_all_trials.append(ensemble_preds)

    cross_model_ensemble_preds = []
    for i, epoch_preds in enumerate(family_preds):
        preds = np.median(np.array(epoch_preds), axis=0)

        cross_model_ensemble_preds.append(preds)

        results['smape']['ens__' + family.name + '____epoch_{}'.format(i)] = np.nanmean(metrics.SMAPE(y, preds[:, -6:]))
        results['mase*']['ens__' + family.name + '____epoch_{}'.format(i)] = np.nanmean(metrics.MASE(x, y, preds[:, -6:]))

    final_preds = np.median(np.array(ensemble_preds_all_trials), axis=0)

    results['smape']['ens__' + family.name + '____epoch_'] = np.nanmean(metrics.SMAPE(y, final_preds[:, -6:]))
    results['mase*']['ens__' + family.name + '____epoch_'] = np.nanmean(metrics.MASE(x, y, final_preds[:, -6:]))

    final_preds_cm = np.median(np.array(cross_model_ensemble_preds), axis=0)

    results['smape']['ens__' + family.name + '__-1__epoch_'] = np.nanmean(metrics.SMAPE(y, final_preds_cm[:, -6:]))
    results['mase*']['ens__' + family.name + '__-1__epoch_'] = np.nanmean(metrics.MASE(x, y, final_preds_cm[:, -6:]))

    return results


@deprecated
def evaluate_family_with_multiple_weights_old(family, x, y, result_dict=None, desc=None, verbose=False):

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

    ensemble_preds_all_trials = []

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

        ensemble_preds_all_trials.append(ensemble_preds)

    for i, epoch_preds in enumerate(family_preds):

        preds = np.median(np.array(epoch_preds), axis=0)

        results['smape']['ens__' + family.name + '____epoch_{}'.format(i)] = np.nanmean(metrics.SMAPE(y, preds[:, -6:]))
        results['mase*']['ens__' + family.name + '____epoch_{}'.format(i)] = np.nanmean(metrics.MASE(x, y, preds[:, -6:]))

    final_preds = np.median(np.array(ensemble_preds_all_trials), axis=0)

    results['smape']['ens__' + family.name + '____epoch_'] = np.nanmean(metrics.SMAPE(y, final_preds[:, -6:]))
    results['mase*']['ens__' + family.name + '____epoch_'] = np.nanmean(metrics.MASE(x, y, final_preds[:, -6:]))

    return results


def evaluate_snapshot_ensemble(family: Union[str, Path], x: np.ndarray, y: np.ndarray, result_dict: dict = None,
                               desc: str = None, batch_size: int = 1024) -> dict:
    """
    Evaluation script for snapshot ensembles

    The function will:

        1) Find all models in the family and all instances of the specific model.
        2) Get the predictiond for each instance.
        3) Compute the sMAPE and MASE estimate of each instance.
        4) Compute the ensemble predictions across instances of a specific model, as well as their sMAPE and MASE*.
        5) Compute the ensemble predictions across different models, per epoch, as well as their sMAPE and MASE*.
        6) Compute the ensemble predictions of the last epoch ensembles of (4), as well as their sMAPE and MASE*.


    The function assumes that the models are saved as:

        results_dir
            |
            |________experiment1__0
            |              |
            |              |________weights_epoch_0.h5
            |              |________weights_epoch_1.h5
            |              |________weights_epoch_2.h5
            |                      ...
            |________experiment1__1
            |               |
            |               |________weights_epoch_0.h5
            |               |________weights_epoch_1.h5
            |               |________weights_epoch_2.h5
            |                        ...
                        ...


    :param family: Location and name of the family of models. In the example structure above, the family would be:
                   'results_dir/experiment'

    :param x: Numpy array containing insample data
    :param y: Numpy array containing out-of-sample data
    :param result_dict: Dictionary to store the reuslts in (optional)
    :param desc: Description for the tqdm (usually the experiment's name)
    :param batch_size: The batch size
    :return: A dictionary containing the sMAPE and MASE* of the individual models and their ensembles.
    """

    if not result_dict:
        results = {'smape': {}, 'mase*': {}}
    else:
        results = result_dict.copy()

    family = Path(family)
    trials = list(family.parent.glob(family.name + '*'))  # TODO: fix sotred bug

    family_preds = []
    num = 0

    for trial in tqdm(trials, desc=desc):

        num += 1
        model_dir = trial / 'best_weights.h5'

        model = tf.keras.models.load_model(model_dir)

        preds = get_predictions(model, x, batch_size=batch_size)
        family_preds.append(preds)

        tf.keras.backend.clear_session()

        results['smape'][family.name + '__' + str(num)] = np.nanmean(metrics.SMAPE(y, preds[:, -6:]))
        results['mase*'][family.name + '__' + str(num)] = np.nanmean(metrics.MASE(x, y, preds[:, -6:]))

        ensemble_preds = np.median(np.array(family_preds), axis=0)

        results['smape'][family.name + '__ens__' + str(num)] = np.nanmean(metrics.SMAPE(y, ensemble_preds[:, -6:]))
        results['mase*'][family.name + '__ens__' + str(num)] = np.nanmean(metrics.MASE(x, y, ensemble_preds[:, -6:]))

    return results


def evaluate_multiple_families(families: Union[str, Sequence], x: np.ndarray, y: np.ndarray, snapshot: bool = False,
                               batch_size: int = 1024):
    """
    Runs single-family evaluation function in a loop for multiple families

    :param families: Sequence of individual families, each containing the location and name of the family of models.
    :param x: Numpy array containing insample data
    :param y: Numpy array containing out-of-sample data
    :param snapshot: Option on whether or not the training used snapshot ensembles
    :param batch_size: The batch size
    :return: A dictionary with the results
    """
    results = {'smape': {}, 'mase*': {}}

    family_eval_fcn = evaluate_snapshot_ensemble if snapshot else evaluate_family_with_multiple_weights

    template = ''
    if isinstance(families, str):
        families = [families]
    else:
        if len(families) > 1:
            num_digits = str(len(str(len(families))))
            template = 'family {:>' + num_digits + '} of {:<' + num_digits + '}'

    for i, family in enumerate(families):
        results = family_eval_fcn(family, x, y, results, desc=template.format(i+1, len(families)), batch_size=batch_size)

        with open('/tmp/{}.pkl'.format(Path(family).name), 'wb') as f:
            pkl.dump(results, f)

    return results


def find_untracked_trials(result_dir: Union[str, Path], tracked: dict = None, exclude_pattern: str = None,
                          verbose: bool = False) -> (dict,) * 3:
    """
    Search for experiments and runs in the 'result_dir', see which of these are already tracked or undertracked and
    return the findings.

    Tracked results examines (a) existence of experiment and (b) number of runs per experiment.

    Glossary:
        - tracked: experiments that are fully tracked (all runs accounted for)
        - untracked: experiments that are completely untracked
        - undertracked: tracked experiments but with fewer runs
        - redundant: experiments that were previously tracked, but whose weights weren't identified

    :param result_dir: Directory that we will search to find runs
    :param tracked: Dictionary that contains what experiments have been tracked and the number of runs per experiment
    :param exclude_pattern: Pattern to exclude from search
    :param verbose: Option to analytically display display results
    :return: dictionaries with untracked, undertracked and redundant experiments
    """

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
                   snapshot=False, debug=False, batch_size=1024, inp_len=18):

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
    
    X_test, y_test = datasets.load_test_set(N=inp_len)

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
        results = evaluate_multiple_families(families, X_test, y_test, snapshot=snapshot, batch_size=batch_size)
        
        if return_results:
            return results, tracked

        results_df_fcn = create_results_df_snapshot if snapshot else create_results_df_multi_weights

        result_df_file = (Path(report_dir) / 'results.csv')
        if result_df_file.exists():
            df = pd.read_csv(result_df_file)
            df = pd.concat([df, results_df_fcn(results, columns=columns)])
        else:
            df = results_df_fcn(results, columns=columns)
        
        if not result_df_file.parent.is_dir():
            os.makedirs(report_dir)

        df.to_csv(result_df_file, index=False)

        with open(tracked_file, 'wb') as f:
            pkl.dump(tracked, f)
