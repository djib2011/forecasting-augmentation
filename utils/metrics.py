import numpy as np


def MASE(x: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Computes the Mean Absolute Scaled Error (MASE) amongst the targets (y) and the predictions (p),
    given the insample data points (x).

    :param x: Array containing insample data points. Should be (num_samples, len_timeseries).
    :param y: Array containing out-of-sample data poitns (i.e. targets). Should be (num_samples, forecast_horizon)
    :param p: Array containing predictions. Should be (num_samples, forecast_horizon)
    :return: The MASE per sample (num_samples,)
    """
    nom = np.mean(np.abs(y - p), axis=1)
    denom = np.mean(np.abs(x[:, 1:] - x[:, :-1]), axis=1) + np.finfo('float').eps
    return nom / denom


def SMAPE(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Computes the symmetric Mean Absolute Percentage Error (sMAPE) amongst the targets (y) and the insample (x)

    :param y: Array containing out-of-sample data poitns (i.e. targets). Should be (num_samples, forecast_horizon)
    :param p: Array containing predictions. Should be (num_samples, forecast_horizon)
    :return: The sMAPE per sample (num_samples,)
    """
    nom = np.abs(y - p)
    denom = np.abs(y) + np.abs(p) + np.finfo('float').eps
    return 2 * np.mean(nom / denom, axis=1) * 100


def OWA(x, y, p):
    """
    Computes the M4 performance metric.

    To compute the Overall Weighted Average between the targets (y) and the predictions (p):
     - sMAPE(y, p) is computed and divided by the sMAPE of a benchmark naive method
     - MASE(x, y, p) is computed and divided by the MASE of a benchmark naive method
     - The average of the two is calculated

    :param x: Array containing insample data points. Should be (num_samples, len_timeseries).
    :param y: Array containing out-of-sample data poitns (i.e. targets). Should be (num_samples, forecast_horizon)
    :param p: Array containing predictions. Should be (num_samples, forecast_horizon)
    :return: The sMAPE per sample (num_samples,)
    """
    rel_smape = SMAPE(y, p) / 15.201
    rel_mase = MASE(x, y, p) / 1.685
    return (rel_smape + rel_mase) / 2
