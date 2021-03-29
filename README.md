# Train NLP models with PTB on TensorFlow # 

## Download and Prepare the Data ##

The data required for this tutorial is in the data/ directory of the PTB dataset from Tomas Mikolov's webpage:
http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

The dataset is already preprocessed and contains overall 10000 different words, including the end-of-sentence marker and a special symbol (<unk>) for rare words. We convert all of them in the reader.py to unique integer identifiers to make it easy for the neural network to process.

