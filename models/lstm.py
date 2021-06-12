from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam
import tensorflow as tf


class LSTMNet:
    def __init__(self, max_length, learn_rate=0.01, optimizer='Adam'):
        self.max_length = max_length
        if optimizer == 'Adam':
            self.opt = tf.keras.optimizers.Adam(learning_rate=learn_rate)
        elif optimizer == 'SGD':
            self.opt = tf.keras.optimizers.SGD(learning_rate=learn_rate)
        elif optimizer == 'Adagrad':
            self.opt = tf.keras.optimizers.Adagrad(learning_rate=learn_rate)
        elif optimizer == 'Momentum':
            self.opt = tf.keras.optimizers.SGD(learning_rate=learn_rate, momentum=0.9)
        else:
            raise ValueError('Optimizer is not recognized')

    def build(self, word2index, tag2index):
        model = Sequential()
        model.add(InputLayer(input_shape=(self.max_length,)))
        model.add(Embedding(len(word2index), 128))
        model.add(LSTM(256, return_sequences=True))
        model.add(LSTM(256, return_sequences=True))
        model.add(TimeDistributed(Dense(len(tag2index))))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=self.opt,
                      metrics=['accuracy'])

        return model
