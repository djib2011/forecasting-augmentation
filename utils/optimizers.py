import tensorflow as tf


def build_optimizer(name, **kwargs):
    """
    Builds an instance of a keras optimizer with the desired hyperparameters

    :param name: Name of the optimizer (one of 'adam', 'rmsprop', 'sgd')
    :param kwargs: Parameters of the optimizer's constructor
    :return:
    """
    if name == 'adam':
        return tf.keras.optimizers.Adam(**kwargs)
    elif name == 'rmsprop':
        return tf.keras.optimizers.Adam(**kwargs)
    elif name == 'sgd':
        return tf.keras.optimizers.SGD(**kwargs)
