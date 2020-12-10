import sys

sys.path.append('.')

import evaluation

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode: Don\'t evaluate any of the models; print lots of diagnostic messages.')

    args = parser.parse_args()

    #result_dir = '.backup/results_4_12_2020/lstm_tuning_windows/'
    result_dir = '/tmp/aug-exp2/results/lstm_tuning_new/'
    report_dir = 'reports/tmp_old_checkout_modified/'

    columns = ['input_len', 'direction', 'size', 'depth']

    evaluation.run_evaluation(result_dir=result_dir, report_dir=report_dir, columns=columns, debug=args.debug, snapshot=False)
