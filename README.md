# Train NLP models on TensorFlow # 

All the models are tested on TensorFlow 1.15.

## LSTM ##

LSTM (Long Short-Term Memory) is a recurrent neural network architecture. The key idea is to use LSTM units partially solve the vanishing gradient problem, since LSTM units allow gradients to also flow unchanged. The similar idea has been applied to convolutional neural network, i.e., residual nerual network. 

The implementation of LSTM is tested on Universal Dependencies Treebank dataset for POS Tagging task.


## BiLSTM ##

BiLSTM (Bidirectional LSTM) is a recurrent neural network architecture that consists of two LSTMs: one taking the input in a forward direction, and the other in a backwards direction. 

The implementation of BiLSTM is tested on Universal Dependencies Treebank dataset for POS Tagging task.


## BERT ##

BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based nerual network architecture for NLP pre-training. 

The implemnetation of BERT is based on the pre-trained model https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1 and tested on Large Movie Review Dataset for Binary Sentiment Classification task.


