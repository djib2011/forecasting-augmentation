import sys

sys.path.append('.')

import evaluation
import datasets


def create_results_df_snap(results):
    df = pd.DataFrame(results).reset_index()
    df['num'] = df['index'].apply(lambda x: x.split('__')[-1])
    df['ens'] = df['index'].apply(lambda x: 'ens' in x)
    return df.drop(columns='index')


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode: Don\'t evaluate any of the models; print lots of diagnostic messages.')

    args = parser.parse_args()

    result_dir = 'results/test_snapshot/'
    report_dir = 'reports/test_snapshot/'

    X_test, y_test = datasets.load_test_set()

    results_with = evaluation.evaluate_snapshot_ensemble(result_dir + 'with_shapshot', X_test, y_test)
    df_with = create_results_df(results_with)

    results_without = evaluation.evaluate_snapshot_ensemble(result_dir + 'no_shapshot', X_test, y_test)
    df_without = create_results_df(results_without)

