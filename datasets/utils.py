import numpy as np
import pandas as pd
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
