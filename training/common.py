import tensorflow as tf
import utils


def train_model(model, train_set, run_name, run_num, cycles=15, batch_size=256):
    result_dir = 'results/'

    epochs = cycles + 5
    callbacks = [utils.callbacks.SnapshotWithAveraging(result_dir + str(run_name), n_cycles=cycles,
                                                       max_epochs=epochs, steps_to_average=100,
                                                       min_warmup_epochs=1, cold_start_id=run_num)]

    model.fit(train_set, epochs=epochs, steps_per_epoch=len(train_set)//batch_size+1, callbacks=callbacks)

    return model
