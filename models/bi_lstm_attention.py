import tensorflow as tf


class AttentionBiLSTM:
    def __init__(self, n_class, voc_size, n_step=2, n_hidden=2, embedding_dim=2):
        self.num_class = n_class
        self.num_step = n_step
        self.num_hidden = n_hidden
        self.voc_size = voc_size
        self.embedding_dim = embedding_dim

    def build(self, train_feature):
        out = tf.Variable(tf.random_normal([self.num_hidden * 2, self.num_class]))

        embedding = tf.Variable(tf.random_uniform([self.voc_size, self.embedding_dim]))

        # [batch_size, len_seq, embedding_dim]
        input = tf.nn.embedding_lookup(embedding, train_feature)

        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.num_hidden)
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.num_hidden)

        output, final_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input, dtype=tf.float32)

        # Attention
        # output[0] : lstm_fw, output[1] : lstm_bw
        output = tf.concat([output[0], output[1]], 2)

        # final_hidden_state : [batch_size, n_hidden * num_directions(=2)]
        final_hidden_state = tf.concat([final_state[1][0], final_state[1][1]], 1)

        # final_hidden_state : [batch_size, n_hidden * num_directions(=2), 1]
        final_hidden_state = tf.expand_dims(final_hidden_state, 2)

        # attn_weights : [batch_size, n_step]
        attn_weights = tf.squeeze(tf.matmul(output, final_hidden_state), 2)

        soft_attn_weights = tf.nn.softmax(attn_weights, 1)

        # context : [batch_size, n_hidden * num_directions(=2), 1]
        context = tf.matmul(tf.transpose(output, [0, 2, 1]), tf.expand_dims(soft_attn_weights, 2))

        # [batch_size, n_hidden * num_directions(=2)]
        context = tf.squeeze(context, 2)

        model = tf.matmul(context, out)

        return model


