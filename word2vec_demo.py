import numpy as np
import tensorflow as tf
from tools.dc_reader import download_dataset, load_data, build_dataset, generate_batch
from tools.model_trainer import train_model, eval_model
from models.word2vec import Word2Vec


def main():
    # Hyper-parameters for training
    batch_size = 128
    embedding_size = 128
    skip_window = 1
    num_skips = 1
    voc_size = 50000
    num_steps = 15001

    # Hyper-parameters for evaluation
    valid_size = 16
    valid_window = 100
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    num_sampled = 64

    # Download the text8 dataset
    dc_url = 'http://mattmahoney.net/dc/'
    text8_file = download_dataset(dc_url, 'text8.zip', 31344016)
    text8_words = load_data(text8_file)
    text8_data, text8_count, text8_dict, text8_reverse_dict = build_dataset(text8_words, 50000)
    del text8_words

    ####################################################
    # Build and train model
    ####################################################

    model = Word2Vec(voc_size, embedding_size)

    feature_ph = tf.placeholder(tf.int32, [batch_size])
    label_ph = tf.placeholder(tf.int32, [batch_size, 1])
    eval_dataset = tf.constant(valid_examples, tf.int32)

    logit = model.build(feature_ph)
    train_op = train_model(logit, label_ph, opt='SGD', lr=0.1)
    eval_op = eval_model(logit, label_ph)

    embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    eval_embeddings = tf.nn.embedding_lookup(normalized_embeddings, eval_dataset)
    similarity = tf.matmul(eval_embeddings, normalized_embeddings, transpose_b=True)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        average_loss = 0

        # count the parameters for the model
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters

        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(text8_data, batch_size, num_skips, skip_window)
            sess.run(train_op, feed_dict={feature_ph: batch_inputs, label_ph: batch_labels})
            step_loss = sess.run(eval_op, feed_dict={feature_ph: batch_inputs, label_ph: batch_labels})
            average_loss += step_loss

            if step + 1 % 1999 == 0:
                average_loss /= 2000
                print("Average loss for step {}: {}".format(step, average_loss))
                average_loss = 0

        sim = similarity.eval()
        for i in range(valid_size):
            valid_word = text8_reverse_dict[valid_examples[i]]
            top_k = 8
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = "most similar word: ".format(str(valid_word))

            for k in range(top_k):
                close_word = text8_reverse_dict[nearest[k]]
                log_str = "%s %s, " % (log_str, close_word)
            print(log_str)


if __name__ == "__main__":
    main()
