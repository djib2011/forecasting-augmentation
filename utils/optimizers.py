import tensorflow as tf


def build_optimizer(name, **kwargs):

    if name == 'adam':
        return tf.keras.optimizers.Adam(**kwargs)
    elif name == 'rmsprop':
        return tf.keras.optimizers.Adam(**kwargs)
    elif name == 'sgd':
        return tf.keras.optimizers.SGD(**kwargs)
