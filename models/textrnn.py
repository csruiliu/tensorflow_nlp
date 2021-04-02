import tensorflow as tf


class TextRNN:
    def __init__(self, n_class, n_step=2, n_hidden=5):
        self.num_step = n_step
        self.num_hidden = n_hidden
        self.num_classes = n_class

    def build(self, train_feature):
        W = tf.Variable(tf.random_normal([self.num_hidden, self.num_classes]))
        b = tf.Variable(tf.random_normal([self.num_classes]))

        cell = tf.nn.rnn_cell.BasicRNNCell(self.num_hidden)
        outputs, states = tf.nn.dynamic_rnn(cell, train_feature, dtype=tf.float32)

        # outputs : [batch_size, n_step, n_hidden]
        # [n_step, batch_size, n_hidden]
        outputs = tf.transpose(outputs, [1, 0, 2])

        # [batch_size, n_hidden]
        outputs = outputs[-1]

        # model : [batch_size, n_class]
        model = tf.matmul(outputs, W) + b

        return model


