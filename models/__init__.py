from . import sequential, sequential_exp


def get(family, type, depth):

    if family == 'sequential':
        return sequential.model_names['{}_{}'.format(type, depth)]


def get_experimental(family, type, depth, property):

    if family == 'sequential':
        return sequential_exp.model_names['{}_{}_{}'.format(type, depth, property)]


def get_optimal_setup(hparams):

    hparams['input_seq_length'] = 18
    hparams['output_seq_length'] = 6
    hparams['base_layer_size'] = 128

    return sequential_exp.model_names['bi_3_batchnorm'], hparams
