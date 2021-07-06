import numpy as np
import tensorflow as tf
from tensorflow import keras

from models.bert import BERT
import tools.lmrd_reader as lmrd_reader


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    keras.backend.set_session(sess)


def main():
    # Params for bert model and tokenization
    bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    max_seq_length = 128

    train_df, test_df = lmrd_reader.download_and_load_datasets()

    # Create datasets (Only take up to max_seq_length words for memory)
    train_text = train_df["sentence"].tolist()
    train_text = [" ".join(t.split()[0:max_seq_length]) for t in train_text]
    train_text = np.array(train_text, dtype=object)[:, np.newaxis]
    train_label = train_df["polarity"].tolist()

    test_text = test_df["sentence"].tolist()
    test_text = [" ".join(t.split()[0:max_seq_length]) for t in test_text]
    test_text = np.array(test_text, dtype=object)[:, np.newaxis]
    test_label = test_df["polarity"].tolist()

    offset = 500

    with tf.Session() as sess:
        # Instantiate tokenizer
        tokenizer = lmrd_reader.create_tokenizer_from_hub_module(bert_path, sess)

        # Convert data to InputExample format
        train_examples = lmrd_reader.convert_text_to_examples(train_text, train_label)
        test_examples = lmrd_reader.convert_text_to_examples(test_text, test_label)

        # Convert to features
        (
            train_input_ids,
            train_input_masks,
            train_segment_ids,
            train_labels,
        ) = lmrd_reader.convert_examples_to_features(tokenizer, train_examples, max_seq_length=max_seq_length)

        (
            test_input_ids,
            test_input_masks,
            test_segment_ids,
            test_labels,
        ) = lmrd_reader.convert_examples_to_features(tokenizer, test_examples, max_seq_length=max_seq_length)

        # 128/2 (bert-tiny), 4/256 (bert-mini), 4/512 (bert-small), 8/512 (bert-medium), 12/768 (bert-base)
        model = BERT(max_seq_length, 128, 2, learn_rate=0.001, optimizer='Adam')
        logit, trainable_parameters = model.build()

        # Instantiate variables
        initialize_vars(sess)

        logit.fit(
            [train_input_ids, train_input_masks, train_segment_ids],
            train_labels,
            epochs=1,
            batch_size=128,
        )

        # scores = logit.evaluate([test_input_ids, test_input_masks, test_segment_ids], test_labels)
        scores = logit.evaluate([test_input_ids[0:offset],
                                 test_input_masks[0:offset],
                                 test_segment_ids[0:offset]],
                                test_labels[0:offset])

        print('{}: {}'.format(logit.metrics_names[1], scores[1]))

        logit.save('bertmodel.h5')

        logit, trainable_parameters = model.build()
        initialize_vars(sess)
        logit.load_weights('bertmodel.h5')

        scores = logit.evaluate([test_input_ids[0:offset],
                                 test_input_masks[0:offset],
                                 test_segment_ids[0:offset]],
                                test_labels[0:offset])

        print('{}: {}'.format(logit.metrics_names[1], scores[1]))


if __name__ == "__main__":
    main()
