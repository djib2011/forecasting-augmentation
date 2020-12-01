from . import sequential


def get(family, type, depth):

    if family == 'sequential':
        return sequential.model_names['{}_{}'.format(type, depth)]
