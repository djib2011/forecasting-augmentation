import numpy as np


def MASE(x, y, p):
    nom = np.mean(np.abs(y - p), axis=1)
    denom = np.mean(np.abs(x[:, 1:] - x[:, :-1]), axis=1) + np.finfo('float').eps
    return nom / denom


def SMAPE(y, p):
    nom = np.abs(y - p)
    denom = np.abs(y) + np.abs(p) + np.finfo('float').eps
    return 2 * np.mean(nom / denom, axis=1) * 100


def OWA(x, y, p):
    rel_smape = SMAPE(y, p) / 15.201
    rel_mase = MASE(x, y, p) / 1.685
    return (rel_smape + rel_mase) / 2
