import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union


def get_last_N(series: Union[pd.Series, np.ndarray], N: int = 18):
    """
    Get the last N points in a timeseries. If len(ts) < N, pad the difference with the first value.

    :param series: A timeseries
    :param N: Number of points to keep
    :return: A timeseries of length N
    """

    ser_N = series.dropna().iloc[-N:].values
    if len(ser_N) < N:
        pad = [ser_N[0]] * (N - len(ser_N))
        ser_N = np.r_[pad, ser_N]
    return ser_N


def load_test_set(data_dir: Union[Path, str] = 'data', N: int = 18) -> (np.ndarray,) * 2:
    """
    Load the yearly M4 test set (both insample and out-of-sample)

    :param data_dir: Path to the directory containing the 'Yearly-train.csv' and 'Yearly-test.csv'
    :param N: Number data points from the insample to load
    :return: Two arrays representing the insample and out-of-sample M4 test set
    """

    train_path = Path(data_dir) / 'Yearly-train.csv'
    test_path = Path(data_dir) / 'Yearly-test.csv'

    train_set = pd.read_csv(train_path).drop('V1', axis=1)
    test_set = pd.read_csv(test_path).drop('V1', axis=1)

    X_test = np.array([get_last_N(ser[1], N=N) for ser in train_set.iterrows()])
    y_test = test_set.values

    return X_test, y_test


def make_combinations(data: np.ndarray, num_samples: int, num_combs: int, seed: int = None) -> np.ndarray:
    """
    Generate timeseries from a given dataset, by combining two or more timesereis.

    :param data: Dataset, based on which, the new timeseries will be generated
    :param num_samples: Number of series to generate
    :param num_combs: Number of original timesereis to combine to generate 1 new timeseries
    :param seed: Random seed - used for reproducability
    :return: Array containing new timeseries
    """

    if seed is not None:
        np.random.seed(seed)

    samples_list = [np.random.choice(np.arange(data.shape[0]), num_samples, replace=True) for _ in range(num_combs)]

    syn = np.array([np.sum([data[s[i]] for s in samples_list], axis=0) / num_combs for i in range(num_samples)])

    return syn


def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Function to normalize a dataset (doesn't fit on last 6 datapoints)

    :param data: Original dataset (should contain both in- and out-of-sample)
    :return: Same dataset, normalized
    """
    mx = data[:, :-6].max(axis=1).reshape(-1, 1)
    mn = data[:, :-6].min(axis=1).reshape(-1, 1)

    return (data - mn) / (mx - mn + np.finfo('float').eps)
