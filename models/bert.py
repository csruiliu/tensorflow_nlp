import tensorflow as tf


class BERT:
    def __init__(self, n_class, n_step=2, n_hidden=2):
        self.num_class = n_class
        self.num_step = n_step
        self.num_hidden = n_hidden

