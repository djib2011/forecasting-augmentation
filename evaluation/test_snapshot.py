import sys

sys.path.append('.')

import evaluation
import datasets

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode: Don\'t evaluate any of the models; print lots of diagnostic messages.')

    args = parser.parse_args()

    result_dir = 'results/test_snapshot/'
    report_dir = 'reports/test_snapshot/'

    X_test, y_test = datasets.load_test_set()

    untracked, _, _ = evaluation.find_untracked_trials(result_dir, exclude_pattern='no_snapshot', verbose=True)
    results_with = evaluation.evaluate_snapshot_ensemble(untracked, X_test, y_test)

    untracked, _, _ = evaluation.find_untracked_trials(result_dir, exclude_pattern='with_snapshot', verbose=True)
    results_without = evaluation.evaluate_snapshot_ensemble(untracked, X_test, y_test)

