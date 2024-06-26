import tensorflow as tf


def bidirectional_2_layer_small_dense(hparams):
    s = hparams['base_layer_size']
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True))(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 2, return_sequences=True))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['output_seq_length'] * s // 2)(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], s // 2))(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inp, out)
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
    return model


def bidirectional_2_layer_dropout(hparams):
    s = hparams['base_layer_size']
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True))(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 2, return_sequences=True))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['output_seq_length'] * s // 2)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], s // 2))(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inp, out)
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
    return model


def bidirectional_2_layer_bn(hparams):
    s = hparams['base_layer_size']
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True))(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 2, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['output_seq_length'] * s // 2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], s // 2))(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inp, out)
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
    return model


def bidirectional_2_layer_ln(hparams):
    s = hparams['base_layer_size']
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True))(inp)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 2, return_sequences=True))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['output_seq_length'] * s // 2)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], s // 2))(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inp, out)
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
    return model


def bidirectional_2_layer_inverse(hparams):
    s = hparams['base_layer_size']
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 2, return_sequences=True))(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['output_seq_length'] * s // 2)(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], s // 2))(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inp, out)
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
    return model


def bidirectional_3_layer_small_dense(hparams):
    s = hparams['base_layer_size']
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True))(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 2, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 4, return_sequences=True))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['output_seq_length'] * s // 2)(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], s // 2))(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inp, out)
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
    return model


def bidirectional_3_layer_bn(hparams):
    s = hparams['base_layer_size']
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True))(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 2, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 4, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['output_seq_length'] * s // 2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], s // 2))(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inp, out)
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
    return model


def bidirectional_3_layer_ln(hparams):
    s = hparams['base_layer_size']
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True))(inp)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 2, return_sequences=True))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 4, return_sequences=True))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['output_seq_length'] * s // 2)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], s // 2))(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inp, out)
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
    return model


def bidirectional_3_layer_dropout(hparams):
    s = hparams['base_layer_size']
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True))(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 2, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 4, return_sequences=True))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['output_seq_length'] * s // 2)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], s // 2))(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inp, out)
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
    return model


def bidirectional_3_layer_inverse(hparams):
    s = hparams['base_layer_size']
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 4, return_sequences=True))(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 2, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['output_seq_length'] * s // 2)(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], s // 2))(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inp, out)
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
    return model


def bidirectional_2_layer_fully_lstm(hparams):
    s = hparams['base_layer_size']
    ratio = hparams['input_seq_length'] // hparams['output_seq_length']

    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True))(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 2, return_sequences=True))(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], ratio * s * 4))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1, return_sequences=True))(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inp, out)
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
    return model


def bidirectional_3_layer_fully_lstm(hparams):
    s = hparams['base_layer_size']
    ratio = hparams['input_seq_length'] // hparams['output_seq_length']

    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True))(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 2, return_sequences=True))(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], ratio * s * 4))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 2, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1, return_sequences=True))(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inp, out)
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
    return model


def bidirectional_2_layer_fully_lstm_bn(hparams):
    s = hparams['base_layer_size']
    ratio = hparams['input_seq_length'] // hparams['output_seq_length']

    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True))(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 2, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], ratio * s * 4))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1, return_sequences=True))(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inp, out)
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
    return model


def bidirectional_3_layer_fully_lstm_bn(hparams):
    s = hparams['base_layer_size']
    ratio = hparams['input_seq_length'] // hparams['output_seq_length']

    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True))(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 2, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], ratio * s * 4))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1, return_sequences=True))(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inp, out)
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
    return model


model_names = {'bi_2_small_dense': bidirectional_2_layer_small_dense,
               'bi_3_small_dense': bidirectional_3_layer_small_dense,
               'bi_2_dropout': bidirectional_2_layer_dropout,
               'bi_3_dropout': bidirectional_3_layer_dropout,
               'bi_2_batchnorm': bidirectional_2_layer_bn,
               'bi_3_batchnorm': bidirectional_3_layer_bn,
               'bi_2_layernorm': bidirectional_2_layer_ln,
               'bi_3_layernorm': bidirectional_3_layer_ln,
               'bi_2_inverse': bidirectional_2_layer_inverse,
               'bi_2_fully': bidirectional_2_layer_fully_lstm,
               'bi_2_fullybn': bidirectional_2_layer_fully_lstm_bn,
               'bi_3_fully': bidirectional_3_layer_fully_lstm,
               'bi_3_fullybn': bidirectional_3_layer_fully_lstm_bn}

if __name__ == '__main__':

    hp = {'base_layer_size': 128,
          'input_seq_length': 18,
          'output_seq_length': 6}

    for name, model_gen in model_names.items():
        model = model_gen(hp)
        model.summary()
