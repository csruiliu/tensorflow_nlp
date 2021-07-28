import tensorflow as tf
import math


class Word2Vec:
    def __init__(self, voc_size, embedding_size):
        self.voc_size = voc_size
        self.embedding_size = embedding_size

    def build(self, train_feature):
        embeddings = tf.Variable(tf.random_uniform([self.voc_size, self.embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_feature)

        nce_weights = tf.Variable(tf.truncated_normal([self.voc_size, self.embedding_size],
                                                      stddev=1.0 / math.sqrt(self.embedding_size)))

        nce_biases = tf.Variable(tf.zeros([self.voc_size]), dtype=tf.float32)

        # W = tf.Variable(tf.random_uniform([self.voc_size, self.embedding_size], -1.0, 1.0))
        # WT = tf.Variable(tf.random_uniform([self.embedding_size, self.voc_size], -1.0, 1.0))
        # [batch_size, embedding_size]
        # hidden_layer = tf.matmul(train_feature, W)
        # [batch_size, voc_size]
        # output_layer = tf.matmul(hidden_layer, WT)

        return embeddings, embed, nce_weights, nce_biases
