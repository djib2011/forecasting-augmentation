from pathlib import Path
import numpy as np
import os
import tensorflow as tf


class SimpleModelCheckpoint(tf.keras.callbacks.Callback):

    def __init__(self, result_file, warmup=0, patience=1, verbose=False, **kwargs):
        """
        Callback for storing the weights of a model every [patience] epochs after a warmup period of [warmup] epochs.

        :param result_file: Where the weights will be stored. This needs to have 'epoch' as a format parameter.
                            For example: '/some/path/weights_{epoch:2d}.h5'
        :param warmup: How many epochs to wait before storing weights.
        :param patience: How many epochs to wait among two consecutive weight storage operations.
        :param verbose: Output message to the user (does not work properly)
        :param kwargs: Does nothing! is used to ignore redundant keyword arguments.
        """
        super().__init__()
        self.result_file = result_file
        self.verbose = verbose
        self.patience = patience
        self.warmup = warmup

    def on_epoch_end(self, epoch, logs=None):

        if epoch == self.warmup or (epoch > self.warmup and epoch % self.patience == 0):
            p = Path(self.result_file.format(epoch=epoch))

            if self.verbose:
                print('\nSaving weights to:', str(p))

            if not p.parent.is_dir():
                os.makedirs(str(p.parent))

            self.model.save(str(p))


class CosineAnnealingLearningRateSchedule(tf.keras.callbacks.Callback):

    def __init__(self, n_epochs, n_cycles, max_lr):
        """
        Callback that creates a cosine-annealing learning rate schedule.

        Learning rate changes every epoch. Schedule looks like this:

        ***               ****               ****               ****               ****               *
           ***            *   ***            *   ***            *   ***            *   ***            *
             ***          *     ***          *     ***          *     ***          *     ***          *
               **         *       **         *       **         *       **         *       **         *
                **        *        **        *        **        *        **        *        **        *
                **        *        **        *        **        *        **        *        **        *
                **        *        **        *        **        *        **        *        **        *
                 **       *         **       *         **       *         **       *         **       *
                  ***     *          ***     *          ***     *          ***     *          ***     *
                    ***   *            ***   *            ***   *            ***   *            ***   *
                       ****               ****               ****               ****               ****

        :param n_epochs: Number of epochs
        :param n_cycles: Number of cycles (one cycle is multiple epochs)
        :param max_lr: Maximum learning rate
        """
        super().__init__()
        self.epochs = n_epochs
        self.cycles = n_cycles
        self.max_lr = max_lr
        self.lrates = list()

    def cosine_annealing(self, epoch, n_epochs, n_cycles, max_lr):
        epochs_per_cycle = np.floor(n_epochs / n_cycles)
        cos_inner = (np.pi * (epoch % epochs_per_cycle)) / epochs_per_cycle
        return max_lr / 2 * (np.cos(cos_inner) + 1)

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.max_lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.lrates.append(lr)


class SnapshotEnsembleOld(tf.keras.callbacks.Callback):
    """
    Implementation of Snapshot Ensembles
    Code taken from: https://machinelearningmastery.com/snapshot-ensemble-deep-learning-neural-network/
    As described in: https://arxiv.org/pdf/1704.00109.pdf
    Cosine annealing: https://arxiv.org/pdf/1608.03983.pdf
    """

    def __init__(self, family_name, n_cycles=10, n_epochs=None, warmup=0, max_lr=0.001, weight_offset=0):
        super().__init__()
        self.cycles = n_cycles
        self.warmup = warmup
        self.offset = (n_cycles - warmup) * weight_offset
        self.epochs = n_epochs if n_epochs else n_cycles
        self.max_lr = max_lr
        self.lrs = []
        self.family_name = family_name

    def cosine_annealing(self, epoch, n_epochs, n_cycles, max_lr):
        epochs_per_cycle = np.floor(n_epochs / n_cycles)
        cos_inner = (np.pi * (epoch % epochs_per_cycle)) / epochs_per_cycle
        return max_lr / 2 * (np.cos(cos_inner) + 1)

    def on_epoch_begin(self, epoch, logs=None):
        if not logs:
            logs = {}
        lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.max_lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.lrs.append(lr)

    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            logs = {}
        epochs_per_cycle = np.floor(self.epochs / self.cycles)
        if (epoch + 1) % epochs_per_cycle == 0 and epoch + 1 > self.warmup:
            run_name = self.family_name + '__{}/best_weights.h5'.format(int(epoch / epochs_per_cycle + self.offset))
            if not os.path.isdir(run_name):
                os.makedirs(run_name)
            self.model.save(run_name)


class SnapshotEnsemble(tf.keras.callbacks.Callback):
    """
    Implementation of Snapshot Ensembles
    As described in: https://arxiv.org/pdf/1704.00109.pdf
    Cosine annealing: https://arxiv.org/pdf/1608.03983.pdf
    """

    def __init__(self, family_name, n_cycles=10, max_epochs=None, cold_start_id=0, min_warmup_epochs=0):

        super().__init__()
        self.cycles = n_cycles
        self.offset = n_cycles * cold_start_id
        self.epochs = max_epochs
        self.min_warmup_epochs = min_warmup_epochs
        self.max_warmup = max_epochs - n_cycles
        self.max_lr = None
        self.lrs = []
        self.family_name = family_name
        self.n_iterations = None
        self.curr_cycle = 0
        self.stagnant = MetricMonitor(steps=100)
        self.in_warmup = True

    def cosine_annealing(self, iteration):
        cos_inner = (np.pi * (iteration % self.n_iterations)) / self.n_iterations
        return self.max_lr / 2 * (np.cos(cos_inner) + 1)

    def warmup_phase(self, logs):
        loss = logs.get('loss')
        self.stagnant(loss)
        if self.stagnant:
            self.in_warmup = False

    def on_train_batch_end(self, batch, logs=None):
        if self.in_warmup:
            if not self.n_iterations:
                self.n_iterations = self.model.history.params['steps']
            if not self.max_lr:
                self.max_lr = self.model.optimizer.learning_rate.numpy()
            self.warmup_phase(logs)
            self.lrs.append(self.max_lr)
        else:
            lr = self.cosine_annealing(iteration=batch)
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
            self.lrs.append(lr)

    def on_epoch_end(self, epoch, logs=None):
        if self.in_warmup:
            if epoch > self.max_warmup:
                self.in_warmup = False
        else:
            if epoch > self.min_warmup_epochs:
                run_dir = self.family_name + '__{}'.format(self.curr_cycle + self.offset)
                if not os.path.isdir(run_dir):
                    os.makedirs(run_dir)
                self.model.save(run_dir + '/best_weights.h5')
                self.curr_cycle += 1
                if self.curr_cycle >= self.cycles:
                    self.model.stop_training = True


class MetricMonitor:

    def __init__(self, steps=10, threshold=0.1):
        self.n = steps
        self.metric = []
        self.full = False
        self.threshold = threshold

    def __call__(self, val):

        self.metric.append(val)

        if self.full:
            self.metric.pop(0)
        else:
            self.full = self.check_full()

    def check_full(self):
        return len(self.metric) >= self.n

    def average(self):
        return np.mean(self.metric)

    def flush(self):
        self.metric = []
        self.full = False

    def no_significant_change(self, threshold=None):

        if not self.full:
            return False

        if not threshold:
            threshold = self.threshold

        threshold = self.average() * (1 + threshold)

        return (np.max(np.abs(self.metric)) - np.min(np.abs(self.metric))) >= threshold

    def __bool__(self):
        return bool(self.no_significant_change())


class SnapshotWithAveraging(SnapshotEnsemble):

    def __init__(self, *args, steps_to_average=100, **kwargs):
        """
        Callback that takes model snapshots of weight averages for ensembling.
        Snapshot ensembles: https://arxiv.org/pdf/1704.00109.pdf
        Cosine annealing: https://arxiv.org/pdf/1608.03983.pdf
        SGDA: https://arxiv.org/pdf/1803.05407.pdf
        :param steps_to_average: How long is the averaging window
        """
        super(SnapshotWithAveraging, self).__init__(*args, **kwargs)
        self.n_average = steps_to_average
        self.average_weights = None

    def add_to_average(self):
        if not self.average_weights:
            self.average_weights = [w / self.n_average for w in self.model.get_weights()]
        else:
            self.average_weights = [m + w / self.n_average for m, w in zip(self.average_weights,
                                                                           self.model.get_weights())]

    def flush_average(self):
        if self.average_weights:
            self.model.set_weights(self.average_weights)
            self.average_weights = None

    def on_train_batch_end(self, batch, logs=None):

        super(SnapshotWithAveraging, self).on_train_batch_end(batch, logs)

        if batch >= self.n_iterations - self.n_average:
            self.add_to_average()

    def on_epoch_end(self, epoch, logs=None):
        curr_weights = self.model.get_weights()
        self.model.set_weights(self.average_weights)
        super(SnapshotWithAveraging, self).on_epoch_end(epoch, logs)
        self.average_weights = curr_weights

    def on_epoch_begin(self, epoch, logs=None):
        self.flush_average()

    @staticmethod
    def compute_statistic(weights):
        if not weights:
            return np.nan
        return np.sum([w.sum() for w in weights])


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # Load training data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32)[..., np.newaxis] / 255
    x_test = x_test.astype(np.float32)[..., np.newaxis] / 255
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Make CNN
    inp = tf.keras.layers.Input((28, 28, 1))
    c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inp)
    c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(c1)
    p2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(c2)
    c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(p2)
    fl = tf.keras.layers.Flatten()(c3)
    out = tf.keras.layers.Dense(10, activation='softmax')(fl)
    model = tf.keras.models.Model(inp, out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 5
    callbacks = [SnapshotWithAveraging('/tmp/snapshot_test/snap', max_epochs=epochs, n_cycles=3)]

    h = model.fit(x_train, y_train, batch_size=256, epochs=epochs, validation_data=(x_test, y_test), callbacks=callbacks)

    plt.plot(callbacks[0].lrs)
    plt.show()
