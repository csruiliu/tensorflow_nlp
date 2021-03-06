from models.bi_lstm import BiLSTM
from models.lstm import LSTMNet
import tools.udtb_reader as udtb_reader
import tensorflow as tf


def main():
    (train_sentences_x,
     val_sentences_x,
     train_tags_y,
     val_tags_y,
     MAX_LENGTH,
     word2index,
     tag2index) = udtb_reader.load_udtb_dataset()

    ####################################################
    # Build model for BiLSTM
    ####################################################

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        model = BiLSTM(MAX_LENGTH, learn_rate=0.001, optimizer='Adam')
        logit, _ = model.build(word2index, tag2index)

        logit.fit(train_sentences_x,
                  udtb_reader.to_categorical(train_tags_y, len(tag2index)),
                  batch_size=64,
                  epochs=1,
                  validation_split=0.2)
        scores = logit.evaluate(val_sentences_x, udtb_reader.to_categorical(val_tags_y, len(tag2index)))
        print(f"{logit.metrics_names[1]}: {scores[1]}")

        logit.save('lstm.h5')

        logit.load_weights('lstm.h5')

        logit.fit(train_sentences_x,
                  udtb_reader.to_categorical(train_tags_y, len(tag2index)),
                  batch_size=64,
                  epochs=1,
                  validation_split=0.2)

        scores = logit.evaluate(val_sentences_x, udtb_reader.to_categorical(val_tags_y, len(tag2index)))
        print(f"{logit.metrics_names[1]}: {scores[1]}")

    ####################################################
    # Build model for LSTM
    ####################################################

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        model = LSTMNet(MAX_LENGTH, learn_rate=0.001, optimizer='Adagrad')
        logit, _ = model.build(word2index, tag2index)

        logit.fit(train_sentences_x,
                  udtb_reader.to_categorical(train_tags_y, len(tag2index)),
                  batch_size=64,
                  epochs=10,
                  validation_split=0.2)
        scores = logit.evaluate(val_sentences_x, udtb_reader.to_categorical(val_tags_y, len(tag2index)))

        print('{}: {}'.format(logit.metrics_names[1], scores[1]))

        logit.save('bilstm.h5')

        logit.load_weights('bilstm.h5')

        logit.fit(train_sentences_x,
                  udtb_reader.to_categorical(train_tags_y, len(tag2index)),
                  batch_size=64,
                  epochs=1,
                  validation_split=0.2)

        scores = logit.evaluate(val_sentences_x, udtb_reader.to_categorical(val_tags_y, len(tag2index)))
        print(f"{logit.metrics_names[1]}: {scores[1]}")


if __name__ == "__main__":
    main()