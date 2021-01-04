import numpy as np
import pandas as pd
import pickle as pkl
import argparse
from pathlib import Path
import sys

sys.path.append('.')

import evaluation

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode: Don\'t evaluate any of the models; print lots of diagnostic messages.')

    args = parser.parse_args()

    result_dir = 'results/comb_nw/'
    report_dir = 'reports/comb_nw/'

    columns = ['input_len', 'n_samples', 'combinations']

    evaluation.run_evaluation(result_dir=result_dir, report_dir=report_dir, columns=columns, debug=args.debug, snapshot=False)

