import tensorflow as tf


class Word2Vec:
    def __init__(self, voc_size, embedding_size=2):
        self.voc_size = voc_size
        self.embedding_size = embedding_size

    def build(self, train_feature):
        # W and WT is not Traspose relationship
        W = tf.Variable(tf.random_uniform([self.voc_size, self.embedding_size], -1.0, 1.0))
        WT = tf.Variable(tf.random_uniform([self.embedding_size, self.voc_size], -1.0, 1.0))

        hidden_layer = tf.matmul(train_feature, W)  # [batch_size, embedding_size]
        output_layer = tf.matmul(hidden_layer, WT)  # [batch_size, voc_size]

        return output_layer
