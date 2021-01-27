"""
Create the dataset that will be used for experiments from the original M4 data.

Command line arguments:
-i: length of the input data. window length = input length + 6
--line: convert the out-of-sample data into a line
--no_window: option to use the last window of the original dataset
             if not used will generate all possible windows of the given length
"""

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle as pkl
from sklearn.preprocessing import MinMaxScaler
import argparse
import warnings
import h5py
import os
import sys

warnings.filterwarnings('ignore')

if os.getcwd().endswith('datasets'):
    os.chdir('..')

sys.path.append(os.getcwd())

from datasets import get_last_N

data_path = Path('data/Yearly-train.csv')

data = pd.read_csv(data_path)
data = data.drop('V1', axis=1)

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_len', type=int, default=12, help='Insample length.')
parser.add_argument('--no_window', action='store_true', help='Don\'t generate windows from the timeseries.')
parser.add_argument('--line', action='store_true', help='Approximate outsample with a linear regression.')

args = parser.parse_args()

window = args.input_len + 6

# Generate line
if args.line:
    from sklearn.linear_model import LinearRegression

    lin_reg = LinearRegression()

    def best_line(y):
        """
        Function that generates a line from data y and returns that line

        :param y: a series
        :return: a line that fits the series
        """
        lin_reg.fit(np.arange(len(y))[:, np.newaxis], y[:, np.newaxis])
        return lin_reg.predict(np.arange(len(y))[:, np.newaxis]).flatten()


def split_series(ser, window):
    """
    Split series into multiple windows.

    :param ser: A series with a length larger than the window
    :param window: Size of the windows that the series will be split into
    :return: An array with a shape of (num_windows, window)
    """
    x = []
    y = []
    for i in range(ser.notna().sum() - window + 1):
        x.append(ser[i:i + window - 6])
        if args.line:
            y.append(best_line(ser[i + window - 6:i + window]))
        else:
            y.append(ser[i + window - 6:i + window])
    return np.array(x), np.array(y)


X, Y = [], []

for s in tqdm(data.values):
    ser = pd.Series(s)

    if args.no_window:
        ser = get_last_N(ser, window)
        X.append(ser[:-6])
        Y.append(ser[-6:])
    else:
        if ser.notna().sum() <= window - 1:
            continue
        x, y = split_series(ser, window)
        X.append(x)
        Y.append(y)

X = np.vstack(X)
Y = np.vstack(Y)

sc = MinMaxScaler()
X = sc.fit_transform(X.T).T
Y = sc.transform(Y.T).T

# Clear outliers
X = X[np.all((Y < 10) & (Y > -10), axis=1)]
Y = Y[np.all((Y < 10) & (Y > -10), axis=1)]

if args.line:
    scale_file = 'data/yearly_{}_scales_line.pkl'.format(window)
    data_file = 'data/yearly_{}_line.h5'.format(window)

elif args.no_window:
    scale_file = 'data/yearly_{}_scales_nw.pkl'.format(window)
    data_file = 'data/yearly_{}_nw.h5'.format(window)

else:
    scale_file = 'data/yearly_{}_scales.pkl'.format(window)
    data_file = 'data/yearly_{}.h5'.format(window)

with open(scale_file, 'wb') as f:
    pkl.dump(scale_file, f)

hf = h5py.File(data_file, 'w')
hf.create_dataset('X', data=X)
hf.create_dataset('y', data=Y)
hf.close()

print('Saved files:')
print(scale_file)
print(data_file)