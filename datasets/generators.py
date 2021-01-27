import numpy as np
import tensorflow as tf
import h5py


def seq2seq_generator(data_path: str, batch_size: int = 256, shuffle: bool = True) -> tf.data.Dataset:
    """
    Factory for building TensorFlow data generators for loading time series data.

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


def artemis_generator(data_path: str, batch_size=256, data_col_start=4, shuffle=True) -> tf.data.Dataset:
    """
    Factory for building TensorFlow data generators for loading time series data that are in Artemis' format.

    Should be used with normalized data.

    :param data_path: Path of a HDF5 file that contains X and y
    :param batch_size: The batch size
    :param data_col_start: Which column to start from (first columns are reserved for IDs, scales, etc.)
    :param shuffle: True/False whether or not the data will be shuffled.
    :return: A TensorFlow data generator.
    """

    # Load the data
    data = np.load(data_path, allow_pickle=True)

    # Split into insample and out-of-sample
    x = data[:, data_col_start:-6]
    y = data[:, -6:]

    # Drop series that have at least one out-of-sample value higher than 10
    x = x[np.all((y < 10) & (y > -10), axis=1)]
    y = y[np.all((y < 10) & (y > -10), axis=1)]

    # Create the tf dataset
    gen = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        gen = gen.shuffle(buffer_size=len(x))
    gen = gen.repeat()
    gen = gen.batch(batch_size=batch_size)
    gen = gen.prefetch(buffer_size=1)

    # Add len attribute
    gen.__class__ = type(gen.__class__.__name__, (gen.__class__,), {'__len__': lambda self: len(x)})

    return gen


if __name__ == '__main__':

    option = 3

    if option == 1:
        import argparse
        
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

    elif option == 2:
        data_path = '/home/artemis/AugmExp/Data/train_in18_all_windows.npy'
        gen = artemis_generator(data_path, batch_size=256, shuffle=True)

        for x, y in gen:
            print('Train set:')
            print(x.shape, y.shape)
            break


    elif option == 3:
        data_path = '/home/artemis/AugmExp/Data/type1/non_m4/train_only_lw_478k_clean.npy'
        gen = artemis_generator(data_path, batch_size=256, shuffle=True)
        print('Gen length: ', len(gen))
        for x, y in gen:
            print('Train set:')
            print(x.shape, y.shape)
            break
