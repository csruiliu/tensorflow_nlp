import pyconll
import os
import urllib
import numpy as np
from keras.preprocessing.sequence import pad_sequences


def read_conllu(path):
    data = pyconll.load_from_file(path)
    tagged_sentences = list()
    t = 0
    for sentence in data:
        tagged_sentence = list()
        for token in sentence:
            if token.upos and token.form:
                t += 1
                tagged_sentence.append((token.form.lower(), token.upos))
        tagged_sentences.append(tagged_sentence)
    return tagged_sentences


def tag_sequence(sentences):
    return [[t for w, t in sentence] for sentence in sentences]


def text_sequence(sentences):
    return [[w for w, t in sentence] for sentence in sentences]


def sentence_split(sentences, max_len):
    new_sentence = list()
    for data in sentences:
        new_sentence.append(([data[x:x + max_len] for x in range(0, len(data), max_len)]))
    new_sentence = [val for sublist in new_sentence for val in sublist]
    return new_sentence


def convert_ner_format(text, label, file):
    with open(file, 'w') as f:
        words = 0
        i = 0
        for zip_i in zip(text, label):
            a, b = tuple(zip_i)
            for r in range(len(a)):
                item = a[r]+' '+b[r]
                f.write("%s\n" % item)
                words += 1
            f.write("\n")
            i += 1
            #if i==3: break
    print('Sentences:', i, 'Words:', words)


def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)


def load_udtb_dataset():
    # Download and load the dataset
    UD_ENGLISH_TRAIN = './dataset/ud_treebank/en_partut-ud-train.conllu'
    UD_ENGLISH_DEV = './dataset/ud_treebank/en_partut-ud-dev.conllu'
    UD_ENGLISH_TEST = './dataset/ud_treebank/en_partut-ud-test.conllu'

    if not os.path.exists(UD_ENGLISH_TRAIN):
        urllib.request.urlretrieve('http://archive.aueb.gr:8085/files/en_partut-ud-train.conllu', UD_ENGLISH_TRAIN)
    if not os.path.exists(UD_ENGLISH_DEV):
        urllib.request.urlretrieve('http://archive.aueb.gr:8085/files/en_partut-ud-dev.conllu', UD_ENGLISH_DEV)
    if not os.path.exists(UD_ENGLISH_TEST):
        urllib.request.urlretrieve('http://archive.aueb.gr:8085/files/en_partut-ud-test.conllu', UD_ENGLISH_TEST)

    train_sentences = read_conllu(UD_ENGLISH_TRAIN)
    val_sentences = read_conllu(UD_ENGLISH_DEV)

    # read the train and eval text
    train_text = text_sequence(train_sentences)
    val_text = text_sequence(val_sentences)

    train_label = tag_sequence(train_sentences)
    val_label = tag_sequence(val_sentences)

    # build dictionary with tag vocabulary
    words, tags = set([]), set([])
    for s in train_text:
        for w in s:
            words.add(w.lower())
    for ts in train_label:
        for t in ts:
            tags.add(t)
    word2index = {w: i + 2 for i, w in enumerate(list(words))}
    word2index['-PAD-'] = 0
    word2index['-OOV-'] = 1
    tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
    tag2index['-PAD-'] = 0

    # prepare the training data
    train_sentences_x, val_sentences_x, train_tags_y, val_tags_y = [], [], [], []
    for s in train_text:
        s_int = []
        for w in s:
            try:
                s_int.append(word2index[w.lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])
        train_sentences_x.append(s_int)

    for s in val_text:
        s_int = []
        for w in s:
            try:
                s_int.append(word2index[w.lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])
        val_sentences_x.append(s_int)

    for s in train_label:
        train_tags_y.append([tag2index[t] for t in s])
    for s in val_label:
        val_tags_y.append([tag2index[t] for t in s])

    MAX_LENGTH = len(max(train_sentences_x, key=len))

    train_sentences_x = pad_sequences(train_sentences_x, maxlen=MAX_LENGTH, padding='post')
    val_sentences_x = pad_sequences(val_sentences_x, maxlen=MAX_LENGTH, padding='post')
    train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
    val_tags_y = pad_sequences(val_tags_y, maxlen=MAX_LENGTH, padding='post')

    return train_sentences_x, val_sentences_x, train_tags_y, val_tags_y, MAX_LENGTH, word2index, tag2index
