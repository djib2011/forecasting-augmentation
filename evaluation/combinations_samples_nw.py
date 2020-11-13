import numpy as np
import pandas as pd
import evaluation

results_dir = 'results/comb_nw/'


def create_results_df(results):

    columns = ['input_len', 'n_samples', 'combinations', 'num']

    single_keys = [k for k in results['smape'].keys() if 'ens' not in k]
    df1 = pd.DataFrame([k.split('__') for k in single_keys], columns=columns)
    df1['ensemble'] = False
    df1['smape'] = [results['smape'][k] if results['smape'][k] else np.nan for k in single_keys]
    df1['mase*'] = [results['mase'][k] if results['mase'][k] else np.nan for k in single_keys]

    ens_keys = [k for k in results['smape'].keys() if 'ens' in k]
    df2 = pd.DataFrame([k.split('__') for k in ens_keys], columns=columns)
    df2['ensemble'] = True
    df2['smape'] = [results['smape'][k] if results['smape'][k] else np.nan for k in ens_keys]
    df2['mase*'] = [results['mase'][k] if results['mase'][k] else np.nan for k in ens_keys]

    df = pd.merge(df1, df2, on=columns)

    for column in columns:
        df[column] = df[column].apply(lambda x: x.split('_')[1])

    return df
