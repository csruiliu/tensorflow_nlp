import tensorflow as tf


class BiLSTM:
    def __init__(self, n_class, n_step=2, n_hidden=2):
        self.num_class = n_class
        self.num_step = n_step
        self.num_hidden = n_hidden

    def build(self, train_feature):
        W = tf.Variable(tf.random_normal([self.num_hidden * 2, self.num_class]))
        b = tf.Variable(tf.random_normal([self.num_class]))

        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.num_hidden)
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.num_hidden)

        # outputs : [batch_size, len_seq, n_hidden], states : [batch_size, n_hidden]
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, train_feature, dtype=tf.float32)

        # output[0] : lstm_fw, output[1] : lstm_bw
        outputs = tf.concat([outputs[0], outputs[1]], 2)

        # [n_step, batch_size, n_hidden]
        outputs = tf.transpose(outputs, [1, 0, 2])

        # [batch_size, n_hidden]
        outputs = outputs[-1]

        model = tf.matmul(outputs, W) + b

        return model

