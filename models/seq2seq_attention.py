import tensorflow as tf


class AttentionSeq2Seq:
    def __init__(self, n_class, n_step=2, n_hidden=2):
        self.num_class = n_class
        self.num_step = n_step
        self.num_hidden = n_hidden

    @staticmethod
    def _get_att_score(dec_output, enc_output, attn):  # enc_output [n_step, n_hidden]
        score = tf.squeeze(tf.matmul(enc_output, attn), 0)  # score : [n_hidden]
        dec_output = tf.squeeze(dec_output, [0, 1])  # dec_output : [n_hidden]
        return tf.tensordot(dec_output, score, 1)  # inner product make scalar value

    def _get_att_weight(self, dec_output, enc_outputs, attn):
        attn_scores = []  # list of attention scalar : [n_step]
        enc_outputs = tf.transpose(enc_outputs, [1, 0, 2])  # enc_outputs : [n_step, batch_size, n_hidden]
        for i in range(self.num_step):
            attn_scores.append(self._get_att_score(dec_output, enc_outputs[i], attn))

        # Normalize scores to weights in range 0 to 1
        return tf.reshape(tf.nn.softmax(attn_scores), [1, 1, -1])  # [1, 1, n_step]

    def build(self, train_feature_encode, train_feature_decode):
        attn = tf.Variable(tf.random_normal([self.num_hidden, self.num_hidden]))
        out = tf.Variable(tf.random_normal([self.num_hidden * 2, self.num_class]))

        model = []
        attention = []

        with tf.variable_scope('encode'):
            enc_cell = tf.nn.rnn_cell.BasicRNNCell(self.num_hidden)
            enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)

            # enc_outputs : [batch_size(=1), n_step(=decoder_step), n_hidden(=128)]
            # enc_hidden : [batch_size(=1), n_hidden(=128)]
            enc_outputs, enc_hidden = tf.nn.dynamic_rnn(enc_cell, train_feature_encode, dtype=tf.float32)

        with tf.variable_scope('decode'):
            dec_cell = tf.nn.rnn_cell.BasicRNNCell(self.num_hidden)
            dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)

            inputs = tf.transpose(train_feature_decode, [1, 0, 2])
            hidden = enc_hidden
            for i in range(self.num_step):
                # time_major True mean inputs shape: [max_time, batch_size, ...]
                dec_output, hidden = tf.nn.dynamic_rnn(dec_cell, tf.expand_dims(inputs[i], 1),
                                                       initial_state=hidden, dtype=tf.float32, time_major=True)

                # attn_weights : [1, 1, n_step]
                attn_weights = self._get_att_weight(dec_output, enc_outputs, attn)
                attention.append(tf.squeeze(attn_weights))

                # matrix-matrix product of matrices [1, 1, n_step] x [1, n_step, n_hidden] = [1, 1, n_hidden]
                context = tf.matmul(attn_weights, enc_outputs)

                # [1, n_step]
                dec_output = tf.squeeze(dec_output, 0)

                # [1, n_hidden]
                context = tf.squeeze(context, 1)

                # [n_step, batch_size(=1), n_class]
                model.append(tf.matmul(tf.concat((dec_output, context), 1), out))

        # to show attention matrix
        trained_attn = tf.stack([attention[0], attention[1], attention[2], attention[3], attention[4]], 0)

        # model : [n_step, n_class]
        model = tf.transpose(model, [1, 0, 2])

        return model
