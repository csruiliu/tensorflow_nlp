import numpy as np

from models.bi_lstm import BiLSTM
from models.lstm import LSTMNet
import tools.udtb_reader as udtb_reader


def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)


if __name__ == "__main__":
    (train_sentences_x,
     val_sentences_x,
     train_tags_y,
     val_tags_y,
     MAX_LENGTH,
     word2index,
     tag2index) = udtb_reader.load_udtb_dataset()

    ####################################################
    # Build model for BiLSTM
    ####################################################

    model = BiLSTM(MAX_LENGTH, learn_rate=0.001, optimizer='Adagrad')
    logit, _ = model.build(word2index, tag2index)

    logit.fit(train_sentences_x,
              to_categorical(train_tags_y, len(tag2index)),
              batch_size=64,
              epochs=10,
              validation_split=0.2)
    scores = logit.evaluate(val_sentences_x, to_categorical(val_tags_y, len(tag2index)))

    print(f"{logit.metrics_names[1]}: {scores[1]}")

    ####################################################
    # Build model for LSTM
    ####################################################

    model = LSTMNet(MAX_LENGTH, learn_rate=0.001, optimizer='Adagrad')
    logit, _ = model.build(word2index, tag2index)

    logit.fit(train_sentences_x,
              to_categorical(train_tags_y, len(tag2index)),
              batch_size=64,
              epochs=10,
              validation_split=0.2)
    scores = logit.evaluate(val_sentences_x, to_categorical(val_tags_y, len(tag2index)))

    print('{}: {}'.format(logit.metrics_names[1], scores[1]))
