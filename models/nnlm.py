import tensorflow as tf


class NNLM:
    def __init__(self, n_class, n_step=2, n_hidden=2):
        self.num_class = n_class
        self.num_step = n_step
        self.num_hidden = n_hidden

    def build(self, train_feature):
        # parameter of H
        H = tf.Variable(tf.random_normal([self.num_step * self.num_class, self.num_hidden]))
        # parameter of d
        d = tf.Variable(tf.random_normal([self.num_hidden]))
        # parameter of d
        U = tf.Variable(tf.random_normal([self.num_hidden, self.num_class]))
        # parameter of b
        b = tf.Variable(tf.random_normal([self.num_class]))

        # [batch_size, n_hidden]
        tanh = tf.nn.tanh(d + tf.matmul(train_feature, H))

        model = tf.matmul(tanh, U) + b  # [batch_size, n_class]

        return model
