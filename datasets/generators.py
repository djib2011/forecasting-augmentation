import numpy as np
import tensorflow as tf
import h5py


def seq2seq_generator(data_path: str, batch_size: int = 256, shuffle: bool = True) -> tf.data.Dataset:
    """
    Factory for building TensorFlow data generators for loading time series data.
    Also supports data augmentation and loading series with overlap for backcast.

    :param data_path: Path of a HDF5 file that contains X and y
    :param batch_size: The batch size
    :param shuffle: True/False whether or not the data will be shuffled.
    :return: A TensorFlow data generator.
    """

    # Load data
    with h5py.File(data_path, 'r') as hf:
        x = np.array(hf.get('X'))
        y = np.array(hf.get('y'))

    x = x[..., np.newaxis]
    y = y[..., np.newaxis]

    # Tensorflow dataset
    data = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        data = data.shuffle(buffer_size=len(x))
    data = data.repeat()
    data = data.batch(batch_size=batch_size)
    data = data.prefetch(buffer_size=1)

    data.__class__ = type(data.__class__.__name__, (data.__class__,), {'__len__': lambda self: len(x)})
    return data


if __name__ == '__main__':

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_len', type=int, default=12, help='Insample length.')
    parser.add_argument('-c', '--combinations', type=int, default=2, help='Number of series combined to produce augmentations.')
    parser.add_argument('-n', '--num_samples', type=int, default=23000, help='Number of augmented samples.')
    parser.add_argument('--no_window', action='store_true', help='Don\'t generate windows from the timeseries.')

    args = parser.parse_args()

    window = args.input_len + 6

    if args.no_window:
        x_path = 'data/yearly_{}_aug_by_{}_num_{}_nw.csv'.format(window, args.combinations, args.num_samples)
        y_path = 'data/yearly_{}_y_nw.csv'.format(window)
    else:
        x_path = 'data/yearly_{}_aug_by_{}_num_{}.csv'.format(window, args.combinations, args.num_samples)
        y_path = 'data/yearly_{}_nw.csv'.format(window)

    gen = seq2seq_generator(x_path, y_path, batch_size=256, shuffle=True)

    for x, y in gen:
        print('Train set:')
        print(x.shape, y.shape)
        break
