import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path
import sys

sys.path.append('.')

import evaluation
import datasets


def create_results_df(results):

    columns = ['input_len', 'n_samples', 'combinations']

    single_keys = [k for k in results['smape'].keys() if 'ens' not in k]
    df1 = pd.DataFrame([k.split('__') for k in single_keys], columns=columns + ['num'])
    df1['ensemble'] = False
    df1['smape'] = [results['smape'][k] if results['smape'][k] else np.nan for k in single_keys]
    df1['mase*'] = [results['mase*'][k] if results['mase*'][k] else np.nan for k in single_keys]

    ens_keys = [k.replace('__ens', '') for k in results['smape'].keys() if 'ens' in k]
    df2 = pd.DataFrame([k.split('__') for k in ens_keys], columns=columns + ['num'])
    df2['ensemble'] = True
    df2['smape'] = [results['smape'][k] if results['smape'][k] else np.nan for k in ens_keys]
    df2['mase*'] = [results['mase*'][k] if results['mase*'][k] else np.nan for k in ens_keys]

    df = pd.concat([df1, df2])

    for column in columns:
        df[column] = df[column].apply(lambda x: x.split('_')[1])

    return df


if __name__ == '__main__':

    results = {'smape': {'inp_18__num_2300000__comb_2__0': 25.44551491449657,
                         'inp_18__num_2300000__comb_2_ens__1': 25.44551491449657,
                         'inp_18__num_2300000__comb_2__1': 24.494396082053754,
                         'inp_18__num_2300000__comb_2_ens__2': 24.889442784856378,
                         'inp_18__num_2300000__comb_2__2': 24.415410741550627,
                         'inp_18__num_2300000__comb_2_ens__3': 24.414445203120817,
                         'inp_18__num_2300000__comb_2__3': 24.826033834016254,
                         'inp_18__num_2300000__comb_2_ens__4': 24.313302818810314,
                         'inp_18__num_230000__comb_2__0': 24.733624216757953,
                         'inp_18__num_230000__comb_2_ens__1': 24.733624216757953,
                         'inp_18__num_230000__comb_2__1': 24.95214180562536,
                         'inp_18__num_230000__comb_2_ens__2': 24.819024318433407,
                         'inp_18__num_230000__comb_2__2': 24.673587783826044,
                         'inp_18__num_230000__comb_2_ens__3': 24.873625890384716,
                         'inp_18__num_230000__comb_2__3': 24.3341261964441,
                         'inp_18__num_230000__comb_2_ens__4': 24.75096721223498},
               'mase*': {'inp_18__num_2300000__comb_2__0': 49.802228562835055,
                         'inp_18__num_2300000__comb_2_ens__1': 49.802228562835055,
                         'inp_18__num_2300000__comb_2__1': 106.47743527116991,
                         'inp_18__num_2300000__comb_2_ens__2': 65.27447817100807,
                         'inp_18__num_2300000__comb_2__2': 70.13376631478,
                         'inp_18__num_2300000__comb_2_ens__3': 63.66254585313159,
                         'inp_18__num_2300000__comb_2__3': 122.94397558355773,
                         'inp_18__num_2300000__comb_2_ens__4': 65.3028413536982,
                         'inp_18__num_230000__comb_2__0': 6.685877705626775,
                         'inp_18__num_230000__comb_2_ens__1': 6.685877705626775,
                         'inp_18__num_230000__comb_2__1': 6.780540625976871,
                         'inp_18__num_230000__comb_2_ens__2': 6.724087953064555,
                         'inp_18__num_230000__comb_2__2': 6.404889746137771,
                         'inp_18__num_230000__comb_2_ens__3': 6.710250994142874,
                         'inp_18__num_230000__comb_2__3': 6.16277140519864,
                         'inp_18__num_230000__comb_2_ens__4': 6.535948603730667}}

    df = create_results_df(results)

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
