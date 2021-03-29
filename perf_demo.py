import tensorflow as tf
import argparse

import tools.ptb_reader as ptb_reader

if __name__ == "__main__":
    ptb_path = "/home/ruiliu/Development/tensorflow_ptb/dataset/"
    ptb_train_data, ptb_valid_data, ptb_test_data, ptb_vocabulary = ptb_reader.ptb_raw_data(ptb_path)
    output = list(ptb_reader.ptb_iterator(ptb_train_data, batch_size=20, num_steps=2))
    print(output[0], output[1])
