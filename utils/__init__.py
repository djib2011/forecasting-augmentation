import warnings
import functools
from utils import callbacks, metrics, optimizers


def deprecated(func):

    @functools.wraps(func)
    def inner(*args, **kwargs):
        warnings.warn('Function "{}" is deprecated!'.format(func.__name__))
        return func(*args, **kwargs)

    return inner
