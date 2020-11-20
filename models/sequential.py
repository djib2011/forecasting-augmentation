import tensorflow as tf


def unidirectional_2_layer(hparams):
    s = hparams['base_layer_size']
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.LSTM(s, return_sequences=True, activation='relu')(inp)
    x = tf.keras.layers.LSTM(s * 2, return_sequences=True, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['output_seq_length'] * s * 2)(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], s * 2))(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inp, out)
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
    return model


def unidirectional_3_layer(hparams):
    s = hparams['base_layer_size']
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.LSTM(s, return_sequences=True, activation='relu')(inp)
    x = tf.keras.layers.LSTM(s * 2, return_sequences=True, activation='relu')(x)
    x = tf.keras.layers.LSTM(s * 4, return_sequences=True, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['output_seq_length'] * s * 2)(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], s * 2))(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inp, out)
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
    return model


def unidirectional_4_layer(hparams):
    s = hparams['base_layer_size']
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.LSTM(s, return_sequences=True, activation='relu')(inp)
    x = tf.keras.layers.LSTM(s * 2, return_sequences=True, activation='relu')(x)
    x = tf.keras.layers.LSTM(s * 4, return_sequences=True, activation='relu')(x)
    x = tf.keras.layers.LSTM(s * 8, return_sequences=True, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['output_seq_length'] * s * 2)(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], s * 2))(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inp, out)
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
    return model


def bidirectional_2_layer(hparams):
    s = hparams['base_layer_size']
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True, activation='relu'))(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 2, return_sequences=True, activation='relu'))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['output_seq_length'] * s * 2)(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], s * 2))(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inp, out)
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
    return model


def bidirectional_3_layer(hparams):
    s = hparams['base_layer_size']
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True, activation='relu'))(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 2, return_sequences=True, activation='relu'))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 4, return_sequences=True, activation='relu'))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['output_seq_length'] * s * 2)(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], s * 2))(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inp, out)
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
    return model


def bidirectional_4_layer(hparams):
    s = hparams['base_layer_size']
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True, activation='relu'))(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 2, return_sequences=True, activation='relu'))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 4, return_sequences=True, activation='relu'))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 8, return_sequences=True, activation='relu'))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['output_seq_length'] * s * 2)(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], s * 2))(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inp, out)
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
    return model


model_names = {'uni_2': unidirectional_2_layer,
               'uni_3': unidirectional_3_layer,
               'uni_4': unidirectional_4_layer,
               'bi_2': bidirectional_2_layer,
               'bi_3': bidirectional_3_layer,
               'bi_4': bidirectional_4_layer}


if __name__ == '__main__':

    hp = {'base_layer_size': 16,
          'input_seq_length': 18,
          'output_seq_length': 8}

    model = bidirectional_2_layer(hp)
    model.summary()
