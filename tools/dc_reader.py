from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import zipfile

import tensorflow as tf
import numpy as np
import os
import urllib
import random
from pathlib import Path
import collections

'''
Data Compression Programs Dataset by Matt Mahoney
http://mattmahoney.net/dc/
'''


def download_dataset(dc_url, file_name, expected_bytes):
    dc_dataset = Path('./dataset/' + file_name)
    if not dc_dataset.exists():
        dc_file, _ = urllib.request.urlretrieve(url=dc_url + file_name, filename=dc_dataset)
    else:
        dc_file = dc_dataset
    statinfo = os.stat(dc_file)
    if statinfo.st_size == expected_bytes:
        print('Found and verified: {}'.format(file_name))
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify {}'.format(file_name))
    return dc_file


def load_data(file_name):
    with zipfile.ZipFile(file_name) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(words, voc_size):
    # build the vocabulary, count the appearance of each word and return the top #voc_size
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(voc_size - 1))
    dictionary = dict()
    # convert word to index, only focus on top #voc_size, so set the other
    for word, _ in count:
        # e.g., {'the': 1, 'UNK': 0, 'a': 12}
        dictionary[word] = len(dictionary)
    data = list()
    # count for word other than top #voc_size
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)

    count[0][1] = unk_count
    reverse_dict = dict(zip(dictionary.values(),dictionary.keys()))

    return data, count, dictionary, reverse_dict


def generate_batch(data, batch_size, num_skips, skip_window):
    '''
    Args:
        batch_size: number of training batch
        num_skips: sample size of each word
        skip_window: the distance for a word can be consider
    Returns:
        every sample in a batch and the associate
    '''

    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= skip_window * 2

    batch = np.ndarray(shape=(batch_size,), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    for i in range(batch_size // num_skips):
        target = skip_window
        targets_avoid = [skip_window]
        for j in range(num_skips):

            while target in targets_avoid:
                target = random.randint(0, span - 1)
            targets_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]

        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels
