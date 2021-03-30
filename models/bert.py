import tensorflow as tf


class NNLM:
    def __init__(self, n_class, n_step=2, n_hidden=2):
        self.num_class = n_class
        self.num_step = n_step
        self.num_hidden = n_hidden

    def build(self, model_input):
        # parameter of H
        H = tf.Variable(tf.random_normal([self.num_step * self.num_class, self.num_hidden]))
        # parameter of d
        d = tf.Variable(tf.random_normal([self.num_hidden]))
        # parameter of d
        U = tf.Variable(tf.random_normal([self.num_hidden, self.num_class]))
        # parameter of b
        b = tf.Variable(tf.random_normal([self.num_class]))

        # [batch_size, n_hidden]
        tanh = tf.nn.tanh(d + tf.matmul(model_input, H))

        model = tf.matmul(tanh, U) + b  # [batch_size, n_class]

        return model

    def train(self, model, train_label, lr=0.001, opt='Adam'):
        train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=train_label))

        if opt == 'Adam':
            train_op = tf.train.AdamOptimizer(lr).minimize(train_loss)
        elif opt == 'SGD':
            train_op = tf.train.GradientDescentOptimizer(lr).minimize(train_loss)
        elif opt == 'Adagrad':
            train_op = tf.train.AdagradOptimizer(lr).minimize(train_loss)
        elif opt == 'Momentum':
            train_op = tf.train.MomentumOptimizer(lr, 0.9).minimize(train_loss)
        else:
            raise ValueError('Optimizer is not recognized')

        return train_op
