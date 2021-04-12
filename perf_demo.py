import tensorflow as tf
import argparse
import sys
import numpy as np
import os

from models.nnlm import NNLM
from models.word2vec import Word2Vec
from models.bi_lstm import BiLSTM
from models.textrnn import TextRNN
from models.transformer import Transformer

import tools.ptb_reader as ptb_reader
import tools.model_trainer as model_trainer


if __name__ == "__main__":

    nlp_model = 'transformer'
    num_steps = 35
    batch_size = 20
    opt = 'Adam'
    lr = 0.0001
    num_epoch = 1
    num_hidden = 2
    embedding_size = 2

    ptb_path = "/home/ruiliu/Development/tensorflow_ptb/dataset/"

    if nlp_model == 'nnlm':
        ptb_train_data, ptb_valid_data, ptb_test_data, ptb_vocabulary_size = ptb_reader.ptb_data_raw(ptb_path)
        train_data = list(ptb_reader.ptb_batch_iterator(ptb_train_data, batch_size, num_steps, ptb_vocabulary_size))
        eval_data = list(ptb_reader.ptb_batch_iterator(ptb_valid_data, batch_size, num_steps, ptb_vocabulary_size))

        num_train_epochs = len(train_data)
        num_eval_epochs = len(eval_data)

        model = NNLM(n_class=ptb_vocabulary_size, n_step=num_steps, n_hidden=num_hidden)
        feature_ph = tf.placeholder(tf.float32, [None, num_steps, ptb_vocabulary_size])
        label_ph = tf.placeholder(tf.float32, [None, ptb_vocabulary_size])
        # build the model
        logit = model.build(feature_ph)

    elif nlp_model == 'bilstm':
        ptb_train_data, ptb_valid_data, ptb_test_data, ptb_vocabulary_size = ptb_reader.ptb_data_raw(ptb_path)
        train_data = list(ptb_reader.ptb_batch_iterator(ptb_train_data, batch_size, num_steps, ptb_vocabulary_size))
        eval_data = list(ptb_reader.ptb_batch_iterator(ptb_valid_data, batch_size, num_steps, ptb_vocabulary_size))

        num_train_epochs = len(train_data)
        num_eval_epochs = len(eval_data)

        model = BiLSTM(n_class=ptb_vocabulary_size, n_step=num_steps, n_hidden=num_hidden)
        feature_ph = tf.placeholder(tf.float32, [None, num_steps, ptb_vocabulary_size])
        label_ph = tf.placeholder(tf.float32, [None, ptb_vocabulary_size])
        # build the model
        logit = model.build(feature_ph)

    elif nlp_model == 'textrnn':
        ptb_train_data, ptb_valid_data, ptb_test_data, ptb_vocabulary_size = ptb_reader.ptb_data_raw(ptb_path)
        train_data = list(ptb_reader.ptb_batch_iterator(ptb_train_data, batch_size, num_steps, ptb_vocabulary_size))
        eval_data = list(ptb_reader.ptb_batch_iterator(ptb_valid_data, batch_size, num_steps, ptb_vocabulary_size))

        num_train_epochs = len(train_data)
        num_eval_epochs = len(eval_data)

        model = TextRNN(n_class=ptb_vocabulary_size, n_step=num_steps, n_hidden=num_hidden)
        feature_ph = tf.placeholder(tf.float32, [None, num_steps, ptb_vocabulary_size])
        label_ph = tf.placeholder(tf.float32, [None, ptb_vocabulary_size])
        # build the model
        logit = model.build(feature_ph)

    elif nlp_model == 'word2vec':
        ptb_train_data, ptb_valid_data, ptb_test_data, ptb_vocabulary_size = ptb_reader.ptb_data_skipgram(ptb_path)

        train_data = list(ptb_reader.ptb_batch_iterator_skipgrams(ptb_train_data, batch_size, ptb_vocabulary_size))
        eval_data = list(ptb_reader.ptb_batch_iterator_skipgrams(ptb_valid_data, batch_size, ptb_vocabulary_size))

        num_train_epochs = len(train_data)
        num_eval_epochs = len(eval_data)

        model = Word2Vec(voc_size=ptb_vocabulary_size, embedding_size=embedding_size)
        feature_ph = tf.placeholder(tf.float32, shape=[None, ptb_vocabulary_size])
        label_ph = tf.placeholder(tf.float32, shape=[None, ptb_vocabulary_size])

        logit = model.build(feature_ph)

    else:
        print('the model is not supported')
        sys.exit()

    # get the train operation
    train_op = model_trainer.train_model(logit, label_ph)

    # get the eval operation
    eval_op = model_trainer.eval_model(logit, label_ph)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(num_train_epochs):
            print("step {} / {}".format(e+1, num_train_epochs))
            train_batch = train_data[e]
            train_input_batch = train_batch[0]
            train_target_batch = train_batch[1]
            sess.run(train_op, feed_dict={feature_ph: train_input_batch, label_ph: train_target_batch})

        sum_accuracy = 0
        for e in range(num_eval_epochs):
            print("evaluation eval {} / {}".format(e+1, num_eval_epochs))
            eval_batch = eval_data[e]
            eval_input_batch = eval_batch[0]
            eval_target_batch = eval_batch[1]
            eval_batch_accuracy = sess.run(eval_op, feed_dict={feature_ph: eval_input_batch, label_ph: eval_target_batch})
            sum_accuracy += eval_batch_accuracy

        avg_accuracy = sum_accuracy / num_eval_epochs

        print(avg_accuracy)