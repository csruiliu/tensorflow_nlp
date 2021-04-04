import tensorflow as tf
import argparse
import numpy as np

from models.nnlm import NNLM

import tools.ptb_reader as ptb_reader
import tools.model_trainer as model_trainer


if __name__ == "__main__":
    ptb_path = "/home/ruiliu/Development/tensorflow_ptb/dataset/"
    ptb_train_data, ptb_valid_data, ptb_test_data, ptb_vocabulary_size = ptb_reader.ptb_raw_data(ptb_path)

    num_steps = 35
    batch_size = 20

    # train_data = list(ptb_reader.ptb_iterator(ptb_train_data, batch_size, num_steps))
    train_data = list(ptb_reader.ptb_batch_iterator(ptb_train_data, batch_size, num_steps, ptb_vocabulary_size))

    eval_data = list(ptb_reader.ptb_batch_iterator(ptb_valid_data, batch_size, num_steps, ptb_vocabulary_size))

    num_train_epochs = len(train_data)
    num_eval_epochs = len(eval_data)

    model = NNLM(n_class=ptb_vocabulary_size, n_step=num_steps, n_hidden=2)
    
    # [batch_size, number of steps, number of Vocabulary]
    feature_ph = tf.placeholder(tf.float32, [None, num_steps, ptb_vocabulary_size])
    label_ph = tf.placeholder(tf.float32, [None, ptb_vocabulary_size])

    input_ph = tf.reshape(feature_ph, shape=[-1, num_steps * ptb_vocabulary_size])

    # build the model
    logit = model.build(input_ph)

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
