from . import sequential, sequential_exp


def get(family, type, depth):

    if family == 'sequential':
        return sequential.model_names['{}_{}'.format(type, depth)]


def get_experimental(family, type, depth, property):

    if family == 'sequential':
        return sequential_exp.model_names['{}_{}_{}'.format(type, depth, property)]
