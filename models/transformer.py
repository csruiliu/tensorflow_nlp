import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np


class Transformer:
    def __init__(self,
                 num_heads=8,
                 dim_embedding=512,
                 dim_feedforward=2048,
                 num_enc_layers=6,
                 num_dec_layers=6,
                 drop_rate=0.1,
                 warmup_steps=400,
                 pos_encoding_type='sinusoid',
                 ls_epsilon=0.1,
                 use_label_smoothing=True,
                 model_name='transformer'):
        """
        Args:
            num_heads [int]: number of heads in multi-head attention unit.
            dim_embedding [int]: dimension of embedding size and the model data flow.
            dim_feedforward [int]: dimension of the feed-forward layer.
            num_encode_layers [int]: number of encoder layers in the encoder.
            num_decode_layers [int]: number of decoder layers in the decoder.
            drop_rate [float]: drop rate in the dropout layer.
            warmup_steps [int]
            pos_encoding_type [str]: type of positional encoding, 'sinusoid' or 'embedding'.
            ls_epsilon [float]: epsilon in the label smoothing function.
            use_label_smoothing [bool]: whether use label smoothing for the truth target.
            model_name [str]: the specific name of a model instance
        """
        self.num_heads = num_heads
        self.dim_embedding = dim_embedding
        self.dim_feedforward = dim_feedforward
        self.num_encode_layers = num_enc_layers
        self.num_decode_layers = num_dec_layers
        self.drop_rate = drop_rate
        self.warmup_steps = warmup_steps
        self.pos_encoding_type = pos_encoding_type
        self.ls_epsilon = ls_epsilon
        self.use_label_smoothing = use_label_smoothing
        self.model_name = model_name

        self.input_id2word = None
        self.target_id2word = None
        self.pad_id = 0
        self.is_training = True

    def _set_is_training(self, training):
        self.is_training = training

    def _embedding(self, inp, vocab_size, zero_pad=True):
        embed_lookup = tf.get_variable("embed_lookup", [vocab_size, self.dim_embedding], tf.float32,
                                       initializer=tf.contrib.layers.xavier_initializer())

        if zero_pad:
            assert self.pad_id == 0
            embed_lookup = tf.concat((tf.zeros(shape=[1, self.dim_embedding]), embed_lookup[1:, :]), 0)

        out = tf.nn.embedding_lookup(embed_lookup, inp)

        return out

    def _positional_encoding_sinusoid(self, inp):
        """
        PE(pos, 2i) = sin(pos / 10000^{2i/d_model})
        PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
        """
        batch, seq_len = inp.shape.as_list()

        with tf.variable_scope('positional_sinusoid'):
            # Copy [0, 1, ..., `inp_size`] by `batch_size` times => matrix [batch, seq_len]
            pos_ind = tf.tile(tf.expand_dims(tf.range(seq_len), 0), [batch, 1])

            # Compute the arguments for sin and cos: pos / 10000^{2i/d_model})
            # Each dimension is sin/cos wave, as a function of the position.
            pos_enc = np.array([
                [pos / np.power(10000., 2. * (i // 2) / self.dim_embedding) for i in range(self.dim_embedding)]
                for pos in range(seq_len)
            ])  # [seq_len, d_model]

            # Apply the cosine to even columns and sin to odds.
            pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])  # dim 2i
            pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])  # dim 2i+1

            # Convert to a tensor
            lookup_table = tf.convert_to_tensor(pos_enc, dtype=tf.float32)  # [seq_len, d_model]
            if True:
                lookup_table = tf.concat((tf.zeros(shape=[1, self.dim_embedding]), lookup_table[1:, :]), 0)

            # [batch, seq_len, d_model]
            out = tf.nn.embedding_lookup(lookup_table, pos_ind)
            return out

    def _positional_encoding_embedding(self, inp):
        batch_size, seq_len = inp.shape.as_list()

        with tf.variable_scope('positional_embedding'):
            # Copy [0, 1, ..., `inp_size`] by `batch_size` times => matrix [batch, seq_len]
            pos_ind = tf.tile(tf.expand_dims(tf.range(seq_len), 0), [batch_size, 1])
            return self._embedding(pos_ind, seq_len, zero_pad=False)  # [batch, seq_len, d_model]

    def _positional_encoding(self, inp):
        if self.pos_encoding_type == 'sinusoid':
            pos_enc = self._positional_encoding_sinusoid(inp)
        else:
            pos_enc = self._positional_encoding_embedding(inp)
        return pos_enc

    def _label_smoothing(self, inp):
        """
        From the paper: "... employed label smoothing of epsilon = 0.1. This hurts perplexity,
        as the model learns to be more unsure, but improves accuracy and BLEU score."
        Args:
            inp (tf.tensor): one-hot encoding vectors, [batch, seq_len, vocab_size]
        """
        vocab_size = inp.shape.as_list()[-1]
        smoothed = (1.0 - self.ls_epsilon) * inp + (self.ls_epsilon / vocab_size)
        return smoothed

    def _construct_padding_mask(self, inp):
        """
        Args: Original input of word ids, shape [batch, seq_len]
        Returns: a mask of shape [batch, seq_len, seq_len], where <pad> is 0 and others are 1s.
        """
        seq_len = inp.shape.as_list()[1]
        mask = tf.cast(tf.not_equal(inp, self.pad_id), tf.float32)
        mask = tf.tile(tf.expand_dims(mask, 1), [1, seq_len, 1])
        return mask

    def _construct_autoregressive_mask(self, target):
        """
        Args: Original target of word ids, shape [batch, seq_len]
        Returns: a mask of shape [batch, seq_len, seq_len].
        """
        batch_size, seq_len = target.shape.as_list()

        tri_matrix = np.zeros((seq_len, seq_len))
        tri_matrix[np.tril_indices(seq_len)] = 1

        mask = tf.convert_to_tensor(tri_matrix, dtype=tf.float32)
        masks = tf.tile(tf.expand_dims(mask, 0), [batch_size, 1, 1])  # copies
        return masks

    def _preprocess(self, inp, inp_vocab, scope):
        # Pre-processing: embedding + positional encoding
        # Output shape: [batch, seq_len, d_model]
        with tf.variable_scope(scope):
            out = self._embedding(inp, inp_vocab, zero_pad=True) + self._positional_encoding(inp)
            out = tf.layers.dropout(out, rate=self.drop_rate, training=self.is_training)

        return out

    def _feed_forward(self, inp, scope='feed-forward'):
        """
        Position-wise fully connected feed-forward network, applied to each position
        separately and identically. It can be implemented as (linear + ReLU + linear) or
        (conv1d + ReLU + conv1d).
        Args:
            inp (tf.tensor): shape [batch, length, d_model]
        """
        out = inp
        with tf.variable_scope(scope):
            # out = tf.layers.dense(out, self.d_ff, activation=tf.nn.relu)
            # out = tf.layers.dropout(out, rate=self.drop_rate, training=self._is_training)
            # out = tf.layers.dense(out, self.d_model, activation=None)

            # by default, use_bias=True
            out = tf.layers.conv1d(out, filters=self.dim_feedforward, kernel_size=1, activation=tf.nn.relu)
            out = tf.layers.conv1d(out, filters=self.dim_embedding, kernel_size=1)

        return out

    def _scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        Args:
            Q (tf.tensor): of shape (h * batch, q_size, d_model)
            K (tf.tensor): of shape (h * batch, k_size, d_model)
            V (tf.tensor): of shape (h * batch, k_size, d_model)
            mask (tf.tensor): of shape (h * batch, q_size, k_size)
        """

        d = self.dim_embedding // self.num_heads
        assert d == q.shape[-1] == k.shape[-1] == v.shape[-1]

        # [h*batch, q_size, k_size]
        out = tf.matmul(q, tf.transpose(k, [0, 2, 1]))
        # scaled by sqrt(d_k)
        out = out / tf.sqrt(tf.cast(d, tf.float32))

        if mask is not None:
            # masking out (0.0) => setting to -inf.
            out = tf.multiply(out, mask) + (1.0 - mask) * (-1e10)

        out = tf.nn.softmax(out)  # [h * batch, q_size, k_size]
        out = tf.layers.dropout(out, training=self.is_training)
        out = tf.matmul(out, v)  # [h * batch, q_size, d_model]

        return out

    def multihead_attention(self, query, memory=None, mask=None, scope='attn'):
        """
        Args:
            query (tf.tensor): of shape (batch, q_size, d_model)
            memory (tf.tensor): of shape (batch, m_size, d_model)
            mask (tf.tensor): shape (batch, q_size, k_size)
        Returns:h
            a tensor of shape (bs, q_size, d_model)
        """
        if memory is None:
            memory = query

        with tf.variable_scope(scope):
            # Linear project to d_model dimension: [batch, q_size/k_size, d_model]
            Q = tf.layers.dense(query, self.dim_embedding, activation=tf.nn.relu)
            K = tf.layers.dense(memory, self.dim_embedding, activation=tf.nn.relu)
            V = tf.layers.dense(memory, self.dim_embedding, activation=tf.nn.relu)

            # Split the matrix to multiple heads and then concatenate to have a larger
            # batch size: [h*batch, q_size/k_size, d_model/num_heads]
            Q_split = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)
            K_split = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)
            V_split = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)
            mask_split = tf.tile(mask, [self.num_heads, 1, 1])

            # Apply scaled dot product attention
            out = self._scaled_dot_product_attention(Q_split, K_split, V_split, mask=mask_split)

            # Merge the multi-head back to the original shape
            out = tf.concat(tf.split(out, self.num_heads, axis=0), axis=2)  # [bs, q_size, d_model]

            # The final linear layer and dropout.
            # out = tf.layers.dense(out, self.d_model)
            # out = tf.layers.dropout(out, rate=self.drop_rate, training=self._is_training)

        return out

    def _encoder_layer(self, inp, input_mask, scope):
        """
        Args:
            inp: tf.tensor of shape (batch, seq_len, embed_size)
            input_mask: tf.tensor of shape (batch, seq_len, seq_len)
        """
        out = inp
        with tf.variable_scope(scope):
            # One multi-head attention + one feed-forward
            out = tc.layers.layer_norm(out + self.multihead_attention(out, mask=input_mask), center=True, scale=True)
            out = tc.layers.layer_norm(out + self._feed_forward(out), center=True, scale=True)
        return out

    def encoder(self, inp, input_mask, scope='encoder'):
        """
        Args:
            inp (tf.tensor): shape (batch, seq_len, embed_size)
            input_mask (tf.tensor): shape (batch, seq_len, seq_len)
            scope (str): name of the variable scope.
        """
        out = inp  # now, (batch, seq_len, embed_size)
        with tf.variable_scope(scope):
            for i in range(self.num_encode_layers):
                out = self._encoder_layer(out, input_mask, f'enc_{i}')
        return out

    def _decoder_layer(self, target, enc_out, input_mask, target_mask, scope):
        out = target
        with tf.variable_scope(scope):
            out = tc.layers.layer_norm(out + self.multihead_attention(out, mask=target_mask, scope='self_attn'),
                                       center=True, scale=True)
            out = tc.layers.layer_norm(out + self.multihead_attention(out, memory=enc_out, mask=input_mask),
                                       center=True, scale=True)
            out = tc.layers.layer_norm(out + self._feed_forward(out))
        return out

    def decoder(self, target, enc_out, input_mask, target_mask, scope='decoder'):
        out = target
        with tf.variable_scope(scope):
            for i in range(self.num_encode_layers):
                out = self._decoder_layer(out, enc_out, input_mask, target_mask, f'dec_{i}')
        return out

    def build(self, train_input, train_target, input_id2word, target_id2word):

        target_vocab = len(target_id2word)
        input_vocab = len(input_id2word)

        with tf.variable_scope(self.model_name):
            # For the input we remove the starting <s> to keep the seq len consistent.
            encode_input = train_input[:, 1:]

            # For the decoder input, we remove the last element, as no more future prediction
            # is gonna be made based on it.
            decode_input = train_target[:, :-1]
            # starts with the first word
            decode_target = train_target[:, 1:]
            decode_target_ohe = tf.one_hot(decode_target, depth=target_vocab)

            if self.use_label_smoothing:
                decode_target_ohe = self._label_smoothing(decode_target_ohe)

            # The input mask only hides the <pad> symbol.
            input_mask = self._construct_padding_mask(encode_input)

            # The target mask hides both <pad> and future words.
            target_mask = self._construct_padding_mask(decode_input)
            target_mask *= self._construct_autoregressive_mask(decode_input)

            # Input embedding + positional encoding
            embedding_input = self._preprocess(encode_input, input_vocab, "input_preprocess")
            encode_out = self.encoder(embedding_input, input_mask)

            # Target embedding + positional encoding
            embedding_decode_input = self._preprocess(decode_input, target_vocab, "target_preprocess")
            decode_out = self.decoder(embedding_decode_input, encode_out, input_mask, target_mask)

            # Make the prediction out of the decoder output.
            # [batch, target_vocab]
            logits = tf.layers.dense(decode_out, target_vocab)
            model = tf.argmax(logits, axis=-1, output_type=tf.int32)

        return model
