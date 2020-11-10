from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle as pkl
from sklearn.preprocessing import MinMaxScaler
import argparse
import warnings
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

if args.line:
    from sklearn.linear_model import LinearRegression

    lin_reg = LinearRegression()

    def best_line(y):
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
    pkl.dump(sc, open('data/yearly_{}_scales_line.pkl'.format(window), 'wb'))
    np.savetxt('data/yearly_{}_X_line.csv'.format(window), X, delimiter=',')
    np.savetxt('data/yearly_{}_y_line.csv'.format(window), Y, delimiter=',')

    print('Saved files:')
    print('data/yearly_{}_scales_line.pkl'.format(window))
    print('data/yearly_{}_X_line.csv'.format(window))
    print('data/yearly_{}_y_line.csv'.format(window))

elif args.no_window:
    pkl.dump(sc, open('data/yearly_{}_scales_nw.pkl'.format(window), 'wb'))
    np.savetxt('data/yearly_{}_X_nw.csv'.format(window), X, delimiter=',')
    np.savetxt('data/yearly_{}_y_nw.csv'.format(window), Y, delimiter=',')

    print('Saved files:')
    print('data/yearly_{}_scales_nw.pkl'.format(window))
    print('data/yearly_{}_X_nw.csv'.format(window))
    print('data/yearly_{}_y_nw.csv'.format(window))

else:
    pkl.dump(sc, open('data/yearly_{}_scales.pkl'.format(window), 'wb'))
    np.savetxt('data/yearly_{}_X.csv'.format(window), X, delimiter=',')
    np.savetxt('data/yearly_{}_y.csv'.format(window), Y, delimiter=',')

    print('Saved files:')
    print('data/yearly_{}_scales.pkl'.format(window))
    print('data/yearly_{}_X.csv'.format(window))
    print('data/yearly_{}_y.csv'.format(window))
