from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.optimizers import Adagrad
from keras import backend as K
import numpy as np


class LSTMNet:
    def __init__(self, max_length, learn_rate=0.01, optimizer='Adam'):
        self.max_length = max_length
        if optimizer == 'Adam':
            self.opt = Adam(lr=learn_rate)
        elif optimizer == 'SGD':
            self.opt = SGD(lr=learn_rate)
        elif optimizer == 'Adagrad':
            self.opt = Adagrad(lr=learn_rate)
        elif optimizer == 'Momentum':
            self.opt = SGD(lr=learn_rate, decay=1e-6, momentum=0.9)
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

        trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))

        model.compile(loss='categorical_crossentropy',
                      optimizer=self.opt,
                      metrics=['accuracy'])

        return model, trainable_count
