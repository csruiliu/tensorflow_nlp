import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences
import urllib

import models.lstm
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

    ####################################################
    # Download and load the dataset
    ####################################################

    UD_ENGLISH_TRAIN = './dataset/ud_treebank/en_partut-ud-train.conllu'
    UD_ENGLISH_DEV = './dataset/ud_treebank/en_partut-ud-dev.conllu'
    UD_ENGLISH_TEST = './dataset/ud_treebank/en_partut-ud-test.conllu'

    if not os.path.exists(UD_ENGLISH_TRAIN):
        urllib.request.urlretrieve('http://archive.aueb.gr:8085/files/en_partut-ud-train.conllu', UD_ENGLISH_TRAIN)
    if not os.path.exists(UD_ENGLISH_DEV):
        urllib.request.urlretrieve('http://archive.aueb.gr:8085/files/en_partut-ud-dev.conllu', UD_ENGLISH_DEV)
    if not os.path.exists(UD_ENGLISH_TEST):
        urllib.request.urlretrieve('http://archive.aueb.gr:8085/files/en_partut-ud-test.conllu', UD_ENGLISH_TEST)

    train_sentences = udtb_reader.read_conllu(UD_ENGLISH_TRAIN)
    val_sentences = udtb_reader.read_conllu(UD_ENGLISH_DEV)
    test_sentences = udtb_reader.read_conllu(UD_ENGLISH_TEST)

    ####################################################
    # Preprocessing
    ####################################################

    train_text = udtb_reader.text_sequence(train_sentences)
    test_text = udtb_reader.text_sequence(test_sentences)
    # val_text = udtb_reader.text_sequence(val_sentences)

    train_label = udtb_reader.tag_sequence(train_sentences)
    test_label = udtb_reader.tag_sequence(test_sentences)
    # val_label = udtb_reader.tag_sequence(val_sentences)

    ####################################################
    # Build dictionary with tag vocabulary
    ####################################################

    words, tags = set([]), set([])

    for s in train_text:
        for w in s:
            words.add(w.lower())

    for ts in train_label:
        for t in ts:
            tags.add(t)

    word2index = {w: i + 2 for i, w in enumerate(list(words))}
    # The special value used for padding
    word2index['-PAD-'] = 0
    # The special value used for OOVs
    word2index['-OOV-'] = 1

    tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
    # The special value used to padding
    tag2index['-PAD-'] = 0

    train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []

    for s in train_text:
        s_int = []
        for w in s:
            try:
                s_int.append(word2index[w.lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])

        train_sentences_X.append(s_int)

    for s in test_text:
        s_int = []
        for w in s:
            try:
                s_int.append(word2index[w.lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])

        test_sentences_X.append(s_int)

    for s in train_label:
        train_tags_y.append([tag2index[t] for t in s])

    for s in test_label:
        test_tags_y.append([tag2index[t] for t in s])

    MAX_LENGTH = len(max(train_sentences_X, key=len))

    train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding='post')
    test_sentences_X = pad_sequences(test_sentences_X, maxlen=MAX_LENGTH, padding='post')
    train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
    test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding='post')

    ####################################################
    # Build model for BiLSTM
    ####################################################

    model = BiLSTM(MAX_LENGTH, learn_rate=0.001, optimizer='Adagrad')
    logit = model.build(word2index, tag2index)

    logit.fit(train_sentences_X,
              to_categorical(train_tags_y, len(tag2index)),
              batch_size=64,
              epochs=10,
              validation_split=0.2)
    scores = logit.evaluate(test_sentences_X, to_categorical(test_tags_y, len(tag2index)))

    print(f"{logit.metrics_names[1]}: {scores[1]}")
    
    ####################################################
    # Build model for LSTM
    ####################################################

    model = LSTMNet(MAX_LENGTH, learn_rate=0.001, optimizer='Adagrad')
    logit = model.build(word2index, tag2index)

    logit.fit(train_sentences_X,
              to_categorical(train_tags_y, len(tag2index)),
              batch_size=64,
              epochs=10,
              validation_split=0.2)
    scores = logit.evaluate(test_sentences_X, to_categorical(test_tags_y, len(tag2index)))

    print('{}: {}'.format(logit.metrics_names[1], scores[1]))
