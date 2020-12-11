from pathlib import Path
import os
import sys
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import pandas as pd

sys.path.append('.')

import evaluation
import datasets

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode: Don\'t evaluate any of the models; print lots of diagnostic messages.')

    args = parser.parse_args()

    result_dir = 'results/test_manual_snapshot/'
    preds_dir = 'predictions/test_manual_snapshot/'

    X_test, y_test = datasets.load_test_set()

    if not os.path.isdir(preds_dir):
        os.makedirs(preds_dir)

    trials = Path(result_dir).glob('*')

    results = {}

    for trial in trials:

        models = list(trial.glob('*.h5'))

        all_preds = []
        trial_ind = 0

        for single_model in tqdm(models):

            model = tf.keras.models.load_model(single_model)

            preds = evaluation.get_predictions(model, X_test)

            all_preds.append(preds)
            np.save('/tmp/' + str(single_model) + '_preds.npy', preds)
            
            del model 
            tf.keras.backend.clear_session()
 
        pd.DataFrame(np.array(all_preds)).to_csv(preds_dir + r + '.csv')


