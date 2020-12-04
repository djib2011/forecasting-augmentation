import sys

sys.path.append('.')

import evaluation

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode: Don\'t evaluate any of the models; print lots of diagnostic messages.')

    args = parser.parse_args()

    result_dir = 'results/lstm_tuning_windows/'
    report_dir = 'reports/lstm_tuning_windows/'

    columns = ['input_len', 'direction', 'size', 'depth']

    evaluation.run_evaluation(result_dir=result_dir, report_dir=report_dir, columns=columns, debug=args.debug, snapshot=False)
