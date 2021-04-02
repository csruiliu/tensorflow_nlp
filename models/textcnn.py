import tensorflow as tf


class TextCNN:
    def __init__(self, voc_size, embedding_size=2, num_classes=2):
        self.voc_size = voc_size
        self.embedding_size = embedding_size
        self.filter_sizes = [2, 2, 2]
        self.num_filters = 3
        self.sequence_length = 3
        self.num_classes = num_classes

    def build(self, train_feature):
        W = tf.Variable(tf.random_uniform([self.voc_size, self.embedding_size], -1.0, 1.0))

        # [batch_size, sequence_length, embedding_size]
        embedded_chars = tf.nn.embedding_lookup(W, train_feature)

        # add channel(=1) [batch_size, sequence_length, embedding_size, 1]
        embedded_chars = tf.expand_dims(embedded_chars, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]))

            conv = tf.nn.conv2d(embedded_chars,  # [batch_size, sequence_length, embedding_size, 1]
                                W,  # [filter_size(n-gram window), embedding_size, 1, num_filters(=3)]
                                strides=[1, 1, 1, 1],
                                padding='VALID')

            h = tf.nn.relu(tf.nn.bias_add(conv, b))
            pooled = tf.nn.max_pool(h,
                                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                    # [batch_size, filter_height, filter_width, channel]
                                    strides=[1, 1, 1, 1],
                                    padding='VALID')

            # dim of pooled : [batch_size(=6), output_height(=1), output_width(=1), channel(=1)]
            pooled_outputs.append(pooled)

        num_filters_total = self.num_filters * len(self.filter_sizes)

        # h_pool : [batch_size(=6), output_height(=1), output_width(=1), channel(=1) * 3]
        h_pool = tf.concat(pooled_outputs, self.num_filters)

        # [batch_size, ]
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        weight = tf.get_variable('W', shape=[num_filters_total, self.num_classes],
                                 initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.Variable(tf.constant(0.1, shape=[self.num_classes]))
        model = tf.nn.xw_plus_b(h_pool_flat, weight, bias)

        return model
