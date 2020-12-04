import numpy as np
import pandas as pd
import pickle as pkl
import argparse
from pathlib import Path
import sys

sys.path.append('.')

import evaluation
import datasets


def create_results_df(results):

    columns = ['input_len']

    single_keys = [k for k in results['smape'].keys() if 'ens' not in k]
    df1 = pd.DataFrame([k.split('__') for k in single_keys], columns=columns + ['real', 'num'])
    df1['ensemble'] = False
    df1['smape'] = [results['smape'][k] if results['smape'][k] else np.nan for k in single_keys]
    df1['mase*'] = [results['mase*'][k] if results['mase*'][k] else np.nan for k in single_keys]

    ens_keys = [k for k in results['smape'].keys() if 'ens' in k]
    df2 = pd.DataFrame([k.replace('__ens', '').split('__') for k in ens_keys], columns=columns + ['real', 'num'])
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


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode: Don\'t evaluate any of the models; print lots of diagnostic messages.')

    args = parser.parse_args()

    result_dir = 'results/benchmark/'
    report_dir = 'reports/benchmark/'

    columns = ['input_len']

    X_test, y_test = datasets.load_test_set()
    results = evaluation.evaluate_family_with_multiple_weights(result_dir + 'inp_18_nw', X_test, y_test, verbose=True)
    df = evaluation.evaluate_family_with_multiple_weights(results, columns=columns)
    df.to_csv(report_dir + 'results.csv', index=False)
