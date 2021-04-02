import tensorflow as tf


class Seq2Seq:
    def __init__(self, n_class, n_step=2, n_hidden=2):
        self.num_class = n_class
        self.num_step = n_step
        self.num_hidden = n_hidden

    def build(self, train_feature_encode, train_feature_decode):
        with tf.variable_scope('encode'):
            enc_cell = tf.nn.rnn_cell.BasicRNNCell(self.num_hidden)
            enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)
            _, enc_states = tf.nn.dynamic_rnn(enc_cell, train_feature_encode, dtype=tf.float32)
            # encoder state will go to decoder initial_state, enc_states : [batch_size, n_hidden(=128)]

        with tf.variable_scope('decode'):
            dec_cell = tf.nn.rnn_cell.BasicRNNCell(self.num_hidden)
            dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)
            outputs, _ = tf.nn.dynamic_rnn(dec_cell, train_feature_decode, initial_state=enc_states, dtype=tf.float32)
            # outputs : [batch_size, max_len+1, n_hidden(=128)]

        # model : [batch_size, max_len+1, n_class]
        model = tf.layers.dense(outputs, self.num_class, activation=None)

        return model
