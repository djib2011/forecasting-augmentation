import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path
import sys

sys.path.append('.')

import evaluation
import datasets


def create_results_df(results):

    columns = ['input_len', 'n_samples', 'combinations', 'num']

    single_keys = [k for k in results['smape'].keys() if 'ens' not in k]
    df1 = pd.DataFrame([k.split('__') for k in single_keys], columns=columns)
    df1['ensemble'] = False
    df1['smape'] = [results['smape'][k] if results['smape'][k] else np.nan for k in single_keys]
    df1['mase*'] = [results['mase*'][k] if results['mase*'][k] else np.nan for k in single_keys]

    ens_keys = [k for k in results['smape'].keys() if 'ens' in k]
    df2 = pd.DataFrame([k.split('__') for k in ens_keys], columns=columns)
    df2['ensemble'] = True
    df2['smape'] = [results['smape'][k] if results['smape'][k] else np.nan for k in ens_keys]
    df2['mase*'] = [results['mase*'][k] if results['mase*'][k] else np.nan for k in ens_keys]

    df = pd.merge(df1, df2, on=columns)

    for column in columns:
        df[column] = df[column].apply(lambda x: x.split('_')[1])

    return df


if __name__ == '__main__':

    result_dir = 'results/comb_nw/'
    report_dir = 'reports/comb_nw/'

    tracked_file = (Path(report_dir) / 'tracked.pkl')
    if tracked_file.exists():
        with open(tracked_file, 'wb') as f:
            tracked = pkl.load(f)
    else:
        tracked = {}

    untracked, undertracked, _ = evaluation.find_untracked_trials(result_dir, tracked, verbose=True)
    untracked.update(undertracked)

    X_test, y_test = datasets.load_test_set()

    families = [Path(result_dir) / u for u in untracked]
    results = evaluation.evaluate_snapshot_ensembles(families, X_test, y_test)

    result_df_file = (Path(report_dir) / 'results.csv')
    if result_df_file.exists():
        df = pd.read_csv(result_df_file)
        df = pd.concat([df, create_results_df(results)])
    else:
        df = create_results_df(results)
