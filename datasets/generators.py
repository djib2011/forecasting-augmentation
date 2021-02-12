import numpy as np
import tensorflow as tf
import h5py


def seq2seq_generator(data_path: str, batch_size: int = 256, shuffle: bool = True,
                      augmentation: float = 0, debug: bool = False) -> tf.data.Dataset:
    """
    Factory for building TensorFlow data generators for loading time series data.

    :param data_path: Path of a HDF5 file that contains X and y
    :param batch_size: The batch size
    :param shuffle: True/False whether or not the data will be shuffled.
    :param augmentation: The percentage of the batch that will be augmented data. E.g. if augmentation == 0.75 and
                         batch_size == 200, then each batch will consist of 50 real series and 150 fake ones.
    :param debug: True/False whether or not to print information about the batches.
    :return: A TensorFlow data generator.
    """

    aug_batch_size = int(batch_size * augmentation)
    real_batch_size = int(batch_size * (1 - augmentation))

    if debug:
        print('---------- Generator ----------')
        print('Augmentation percentage:', augmentation)
        print('Batch size:             ', batch_size)
        print('Real batch size:        ', real_batch_size)
        print('Augmentation batch size:', aug_batch_size)
        print('Max aug num:            ', real_batch_size * (real_batch_size - 1) // 2)
        print('------------------------------')

    def augment(x, y):
        random_ind_1 = tf.random.categorical(tf.math.log([[1.] * real_batch_size]), aug_batch_size)
        random_ind_2 = tf.random.categorical(tf.math.log([[1.] * real_batch_size]), aug_batch_size)

        x_aug = (tf.gather(x, random_ind_1) + tf.gather(x, random_ind_2)) / 2
        y_aug = (tf.gather(y, random_ind_1) + tf.gather(y, random_ind_2)) / 2

        return tf.concat([x, tf.squeeze(x_aug, [0])], axis=0), tf.concat([y, tf.squeeze(y_aug, [0])], axis=0)

    # Load data
    with h5py.File(data_path, 'r') as hf:
        x = np.array(hf.get('X'))
        y = np.array(hf.get('y'))

    x = x[..., np.newaxis]
    y = y[..., np.newaxis]

    # TensorFlow Dataset
    data = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        data = data.shuffle(buffer_size=len(x))
    data = data.repeat()
    data = data.batch(batch_size=real_batch_size)
    if augmentation:
        data = data.map(augment)
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


def artemis_online_interpolation_generator(data_path, batch_size):

    def augment_interpolate(x, y):
        # Select random subsample from the time series
        max_idx = sample_size - win_size
        start_id = tf.random.uniform((), minval=0, maxval=max_idx, dtype=tf.dtypes.int64)
        slice_idx = tf.range(0, win_size, dtype=tf.dtypes.int64)
        temp_ts = tf.concat([x, y], axis=0)
        subsample = tf.gather(temp_ts, slice_idx + start_id)

        # Interpolate to fill middle points
        xd = tf.range(((win_size * 2) - 1), dtype=np.float64, delta=1)
        new_ts = interp_regular_1d_grid(xd, x_ref_min=0, x_ref_max=24, y_ref=subsample)
        new_ts = tf.gather(new_ts, tf.range(1, (sample_size + 1), dtype=tf.dtypes.int64))

        # Split to input and output tensors
        new_x = tf.gather(new_ts, tf.range(0, 18, dtype=tf.dtypes.int64))
        new_y = tf.gather(new_ts, tf.range(18, 24, dtype=tf.dtypes.int64))

        # Scale new sample
        max_x = tf.math.reduce_max(new_x)
        min_x = tf.math.reduce_min(new_x)
        new_x = (new_x - min_x) / (max_x - min_x)
        new_y = (new_y - min_x) / (max_x - min_x)

        # return new_x, new_y
        if ((tf.reduce_max(new_y) < 10) and (tf.reduce_min(new_y) > -10)):
            return new_x, new_y
        else:
            return x, y

    # Params
    win_size = 13
    sample_size = 24

    # Load preprocessed data
    temp = np.load(data_path, allow_pickle=True)
    x_t = temp[:, 4:22]                             # SHOULD CHANGE DEPENDING ON THE INPUT FILE
    y_t = temp[:, -6:]

    # Clear outliers
    x_t = x_t[np.all((y_t < 10) & (y_t > -10), axis=1)]
    y_t = y_t[np.all((y_t < 10) & (y_t > -10), axis=1)]
    train_length = len(x_t) * 2

    # Training data
    data_train = tf.data.Dataset.from_tensor_slices((x_t, y_t))
    data_train_aug = data_train.map(augment_interpolate, num_parallel_calls=tf.data.AUTOTUNE)
    data_train = data_train.concatenate(data_train_aug)
    data_train = data_train.shuffle(buffer_size=train_length)
    data_train = data_train.repeat()
    data_train = data_train.batch(batch_size=batch_size)
    data_train = data_train.prefetch(buffer_size=tf.data.AUTOTUNE)

    gen.__class__ = type(gen.__class__.__name__, (gen.__class__,), {'__len__': lambda self: train_length})

    return data_train


if __name__ == '__main__':

    option = 4

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

    elif option == 4:
        data_path = '/home/artemis/AugmExp/Data/type1/train_in18_win1000.npy'
        gen = artemis_generator(data_path, batch_size=256, shuffle=True)
        print('Gen length: ', len(gen))
        for x, y in gen:
            print('Train set:')
            print(x.shape, y.shape)
            break
