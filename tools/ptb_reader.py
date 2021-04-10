from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import collections


def _read_words(filename):
    with tf.io.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data]


def ptb_data_raw(folder_path):
    train_path = os.path.join(folder_path, "ptb.train.txt")
    valid_path = os.path.join(folder_path, "ptb.valid.txt")
    test_path = os.path.join(folder_path, "ptb.test.txt")

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary_size = len(word_to_id)

    return train_data, valid_data, test_data, vocabulary_size


def ptb_data_skipgram(folder_path, skip_grams=1):
    train_path = os.path.join(folder_path, "ptb.train.txt")
    valid_path = os.path.join(folder_path, "ptb.valid.txt")
    test_path = os.path.join(folder_path, "ptb.test.txt")

    train_word_sequence = _read_words(train_path)
    valid_word_sequence = _read_words(valid_path)
    test_word_sequence = _read_words(test_path)

    word_to_id = _build_vocab(train_path)

    vocabulary_size = len(word_to_id)

    train_data_skipgrams = list()
    valid_data_skipgrams = list()
    test_data_skipgrams = list()
    for i in range(skip_grams, vocabulary_size - skip_grams):
        train_target = word_to_id[train_word_sequence[i]]
        valid_target = word_to_id[valid_word_sequence[i]]
        test_target = word_to_id[test_word_sequence[i]]

        train_context = [word_to_id[train_word_sequence[i - skip_grams]],
                         word_to_id[train_word_sequence[i + skip_grams]]]
        valid_context = [word_to_id[valid_word_sequence[i - skip_grams]],
                         word_to_id[valid_word_sequence[i + skip_grams]]]
        test_context = [word_to_id[test_word_sequence[i - skip_grams]],
                        word_to_id[test_word_sequence[i + skip_grams]]]

        for ct in train_context:
            train_data_skipgrams.append([train_target, ct])

        for ct in valid_context:
            valid_data_skipgrams.append([valid_target, ct])

        for ct in test_context:
            test_data_skipgrams.append([test_target, ct])

    return train_data_skipgrams, valid_data_skipgrams, test_data_skipgrams, vocabulary_size


def ptb_iterator(raw_data, batch_size, num_steps):
    """Iterate on the raw PTB data.

    This generates batch_size pointers into the raw PTB data,
    and allows minibatch iteration along these pointers.

    Args:
        raw_data: one of the raw data outputs from ptb_raw_data.
        batch_size: int, the batch size, the total number of training examples present in a single batch.
        num_steps: int, the number of unrolls for models like LSTM.
    Yields:
        Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
        The second element of the tuple is the same data time-shifted to the
        right by one.
    Raises:
        ValueError: if batch_size or num_steps are too high.
    """
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)

    # the number of batches
    batch_num = data_len // batch_size

    data = np.zeros([batch_size, batch_num], dtype=np.int32)

    for i in range(batch_size):
        data[i] = raw_data[batch_num * i:batch_num * (i + 1)]

    # the second element is the same data time-shifted to the right by one, so "batch_num - 1"
    # according to num of unrolled step, we get the number of iterations in a single epoch
    epoch_size = (batch_num - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield x, y


def ptb_batch_iterator(raw_data, batch_size, num_steps, vocab_size):
    raw_data = np.array(raw_data, dtype=np.int32)
    data_len = len(raw_data)
    batch_num = data_len // batch_size

    data = np.zeros([batch_size, batch_num], dtype=np.int32)

    for i in range(batch_size):
        data[i] = raw_data[batch_num * i:batch_num * (i + 1)]

    epoch_size = (batch_num - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        x_one_hot = np.zeros((batch_size, num_steps, vocab_size))
        for j in range(batch_size):
            x_one_hot[j, np.arange(num_steps), x[j]] = 1

        y = data[:, (i + 1) * num_steps + 1]
        y_one_hot = np.zeros((batch_size, vocab_size))
        for k in range(batch_size):
            y_one_hot[k, y[k]] = 1

        yield x_one_hot, y_one_hot


def ptb_batch_iterator_skipgrams(raw_data, batch_size, vocab_size):
    for i in np.arange(0, len(raw_data), batch_size):
        random_input = np.eye(vocab_size)[raw_data[i][0]]
        random_label = np.eye(vocab_size)[raw_data[i][1]]
        yield random_input, random_label