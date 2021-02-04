from . import sequential, sequential_exp
from typing import Callable


def get(family: str, type: str, depth: str) -> Callable:
    """
    Returns a model factory from 'sequential' based on its name and depth.

    :param family: Name of the family of the model (only 'sequential' models accepted)
    :param type: Type of the model (i.e. 'uni' or 'bi')
    :param depth: Depth of the model (2, 3, or 4)
    :return: a factory for creating the model with the desired properties
    """

    if family == 'sequential':
        return sequential.model_names['{}_{}'.format(type, depth)]


def get_experimental(family: str, type: str, depth: str, property: str) -> Callable:
    """
    Returns a model factory from 'sequential_exp' based on its name and depth and property.

    :param family: Name of the family of the model (only 'sequential' models accepted)
    :param type: Type of the model (i.e. 'uni' or 'bi')
    :param depth: Depth of the model (2, 3, or 4)
    :param property: Property of the model ('small_dense', 'dropout', 'batchnorm', 'layernorm',
                                            'inverse', 'fully' or 'fullybn')
    :return: a factory for creating the model with the desired properties
    """

    if family == 'sequential':
        return sequential_exp.model_names['{}_{}_{}'.format(type, depth, property)]


def get_optimal_setup(hparams: dict = None) -> Callable:
    """
    Returns a factory for creating the best model.

    :param hparams: a dictionary with the desired hyperparameters. Will be overwritten to match the best hparams.
    :return: factory for creating the best model
    """

    if hparams is None:
        hparams = {}

    hparams['input_seq_length'] = 18
    hparams['output_seq_length'] = 6
    hparams['base_layer_size'] = 256

    return sequential_exp.model_names['bi_2_fully'], hparams

