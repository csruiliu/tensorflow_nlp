from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam


class BiLSTM:
    def __init__(self, max_length):
        self.max_length = max_length

    def build(self, word2index, tag2index):
        model = Sequential()
        model.add(InputLayer(input_shape=(self.max_length,)))
        model.add(Embedding(len(word2index), 128))
        model.add(Bidirectional(LSTM(256, return_sequences=True)))
        model.add(TimeDistributed(Dense(len(tag2index))))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(0.001),
                      metrics=['accuracy'])

        return model
