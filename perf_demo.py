import tensorflow as tf
import argparse
import sys
import os
import numpy as np

from models.nnlm import NNLM
from models.word2vec import Word2Vec

import tools.ptb_reader as ptb_reader
import tools.model_trainer as model_trainer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', action='store', type=str, default='nnlm',
                        help='the name of nlp model')
    parser.add_argument('-u', '--unroll_step', action='store', type=int, default=35,
                        help='the num of unrolled step for a nlp model')
    parser.add_argument('-b', '--batchsize', action='store', type=int, default=20,
                        help='set the training batch size')
    parser.add_argument('-o', '--opt', action='store', type=str,
                        choices=['Adam', 'SGD', 'Adagrad', 'Momentum'],
                        help='set the training optimizer')
    parser.add_argument('-r', '--lr', action='store', type=float,
                        help='set the learning rate')
    parser.add_argument('-e', '--epoch', action='store', type=int,
                        help='set the number of training epoch')

    args = parser.parse_args()
    nlp_model = args.model
    num_steps = args.unroll_step
    batch_size = args.batchsize
    opt = args.opt
    lr = args.lr
    num_epoch = args.epoch

    ptb_path = "/home/ruiliu/Development/tensorflow_ptb/dataset/"

    if nlp_model == 'nnlm':
        ptb_train_data, ptb_valid_data, ptb_test_data, ptb_vocabulary_size = ptb_reader.ptb_data_raw(ptb_path)
        train_data = list(ptb_reader.ptb_batch_iterator(ptb_train_data, batch_size, num_steps, ptb_vocabulary_size))
        eval_data = list(ptb_reader.ptb_batch_iterator(ptb_valid_data, batch_size, num_steps, ptb_vocabulary_size))

        num_train_epochs = len(train_data)
        num_eval_epochs = len(eval_data)

        model = NNLM(n_class=ptb_vocabulary_size, n_step=35, n_hidden=2)
        feature_ph = tf.placeholder(tf.float32, [None, num_steps, ptb_vocabulary_size])
        label_ph = tf.placeholder(tf.float32, [None, ptb_vocabulary_size])

    elif nlp_model == 'word2vec':
        ptb_train_data, ptb_valid_data, ptb_test_data, ptb_vocabulary_size = ptb_reader.ptb_data_skipgram(ptb_path)

        train_data = list(ptb_reader.ptb_batch_iterator_skipgrams(ptb_train_data, batch_size, ptb_vocabulary_size))
        eval_data = list(ptb_reader.ptb_batch_iterator_skipgrams(ptb_valid_data, batch_size, ptb_vocabulary_size))

        num_train_epochs = len(train_data)
        num_eval_epochs = len(eval_data)

        model = Word2Vec(voc_size=ptb_vocabulary_size, embedding_size=2)
        feature_ph = tf.placeholder(tf.float32, shape=[None, ptb_vocabulary_size])
        label_ph = tf.placeholder(tf.float32, shape=[None, ptb_vocabulary_size])

    else:
        print('the model is not supported')
        sys.exit()

    logit = model.build(feature_ph)

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