import tensorflow as tf
import itertools
import utils


def train_model_snapshot(model, train_set, run_name, run_num, cycles=15, batch_size=256):
    result_dir = 'results/'

    epochs = cycles + 5
    callbacks = [utils.callbacks.SnapshotWithAveraging(result_dir + str(run_name), n_cycles=cycles,
                                                       max_epochs=epochs, steps_to_average=100,
                                                       min_warmup_epochs=1, cold_start_id=run_num)]

    model.fit(train_set, epochs=epochs, steps_per_epoch=len(train_set)//batch_size+1, callbacks=callbacks)

    return model


<<<<<<< HEAD
def train_model_single(model, train_set, run_name, run_num, cycles=15, batch_size=256):

    result_file = 'results/{}__{}/'.format(run_name, run_num) + 'weights_epoch_{epoch:02d}.h5'

    epochs = cycles
    callbacks = [tf.keras.callbacks.ModelCheckpoint(result_file, monitor='loss', verbose=1,
                                                    save_best_only=False, period=1)]

    model.fit(train_set, epochs=epochs, steps_per_epoch=len(train_set)//batch_size+1, callbacks=callbacks)

    return model
=======
def make_runs(hparam_combinations_dict):

    names = hparam_combinations_dict.keys()
    combs = itertools.product(*hparam_combinations_dict.values())

    for c in combs:
        yield dict(zip(names, c))
>>>>>>> d5aeece2749439e810bfa0d4f475fce3ffa31e4a
