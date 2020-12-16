import sys

sys.path.append('.')

import evaluation

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode: Don\'t evaluate any of the models; print lots of diagnostic messages.')

    args = parser.parse_args()

    result_dir = 'results/windows/'
    report_dir = 'reports/windows/'

    columns = ['input_len', 'num_samples']

    evaluation.run_evaluation(result_dir=result_dir, report_dir=report_dir, columns=columns, debug=args.debug,
                              snapshot=False)
