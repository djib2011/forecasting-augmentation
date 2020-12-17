import sys

sys.path.append('.')

import evaluation

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode: Don\'t evaluate any of the models; print lots of diagnostic messages.')

    args = parser.parse_args()

    result_dir = 'results/training_hyperparameters/'
    report_dir = 'reports/training_hyperparameters/'

    columns = ['optimizer', 'learning_rate', 'amsgrad', 'exponential_decay']

    evaluation.run_evaluation(result_dir=result_dir, report_dir=report_dir, columns=columns, debug=args.debug, snapshot=False)
