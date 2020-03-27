Introduction to NLP with Deep Learning
======================================

### 1) Introduction to NLP and Word Embeddings

The first notebook introduces some of the fundamental concepts of Natural Language Processing with an emphasis on the aspects that are relevant to Deep Learning. It introduces the concepts of language modeling, N-grams, vector semantics, word embeddings, etc.

Notebook:
([Jupyter nbviewer](https://nbviewer.jupyter.org/github/ccarpenterg/introNLP/blob/master/01a_intro_NLP_and_word_embeddings.ipynb))
([github](https://github.com/ccarpenterg/introNLP/blob/master/01a_intro_NLP_and_word_embeddings.ipynb))

### 2a) Natural Language Processing and Neural Networks

In this notebook, we build and train a shallow neural network for the sentiment analysis of the IMDB Reviews Dataset. Since regular neural networks are not designed for processing sequences of variable length, we use the Gloval Average Pooling to convert sequence of word embedding vectors into one vector. In this manner, we turn every IMDB review into a vector that represents an embedding for the entire sequence. This vector is then fed to a neural network with three layers: input layer, hidden layer and output layer.

The last layer contains only one neuron, and it represents the probability for a particular IMDB review to be positive. This last layer uses sigmoid as the activation function.

Our model is trained as an end-to-end neural network, meaning that the network learns both the word embeddings and the layers' weights at the same time.

Notebook:
([Jupyter nbviewer](https://nbviewer.jupyter.org/github/ccarpenterg/introNLP/blob/master/02a_NLP_and_neural_networks.ipynb))
([github](https://github.com/ccarpenterg/introNLP/blob/master/02a_NLP_and_neural_networks.ipynb))

### 2b) Pretrained Sentence Embeddings and Neural Networks

In the previous notebook, we trained an embedding layer from scratch using a subwords representation of around 8,000 tokens. Now we will be working with pretrained sentence embeddings that have been trained on the Google News dataset (200 billion words).

Each IMDB review is fed to the pretrained sentence embedding layer and subsequently converted into a vector. This vector is the processed by a shallow neural network of 3 layers: input layer, hidden layer and output layer.

We use two different pretrained sentence embeddings and try two different approaches for training our neural network. In one case we freeze our pretrained sentence embeddings' weights and in the other, we fine-tune them.

Notebook:
([Jupyter nbviewer](https://nbviewer.jupyter.org/github/ccarpenterg/introNLP/blob/master/02b_NLP_and_neural_networks.ipynb))
([github](https://github.com/ccarpenterg/introNLP/blob/master/02b_NLP_and_neural_networks.ipynb))

### 3a) Natural Language Processing and Recurrent Neural Networks

We now introduce the Recurrent Neural Networks that are capable of processing sequences of any length

Notebook:
([Jupyter nbviewer](https://nbviewer.jupyter.org/github/ccarpenterg/introNLP/blob/master/03a_NLP_and_recurrent_neural_networks.ipynb))
([github](https://github.com/ccarpenterg/introNLP/blob/master/03a_NLP_and_recurrent_neural_networks.ipynb))

### 3b) Pretrained Word Embedding and Recurrent Neural Networks

In this section, we train 4 different RNNs for the taks of sentiment analysis

I. RNN with 1 bidirectional LSTM layer and word embeding layer trained from scratch

II. RNN with 1 bidirectional LSTM layer and a fine-tuned pretrained word embedding layer (GloVe)

III. RNN with 1 bidirectional LSTM layer and a frozen word embedding layer (Glove)

Notebook:
([Jupyter nbviewer](https://nbviewer.jupyter.org/github/ccarpenterg/introNLP/blob/master/03b_NLP_and_recurrent_neural_networks.ipynb))
([github](https://github.com/ccarpenterg/introNLP/blob/master/03b_NLP_and_recurrent_neural_networks.ipynb))

### 4a) Neural Machine Translation with a Sequence-to-Sequence Model

Notebook:
([Jupyter nbviewer](https://nbviewer.jupyter.org/github/ccarpenterg/introNLP/blob/master/04a_NLP_and_sequence_to_sequence_RNNs.ipynb))
([github](https://github.com/ccarpenterg/introNLP/blob/master/04a_NLP_and_sequence_to_sequence_RNNs.ipynb))

### 4b) Neural Machine Translation with a Seq2Seq and Attention Model

## References

**Lectures and Presentations**

[1] Introduction and Word Vectors [(slides)](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture01-wordvecs1.pdf) [(video)](https://youtu.be/8rXD5-xhemo). CS224n: NLP with Deep Learning (2019), Stanford University

[2] Word Vectors 2 and Word Senses [(slides)](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture02-wordvecs2.pdf) [(video)](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/cs224n-2019-notes02-wordvecs2.pdf). CS224n: NLP with Deep Learning (2019), Stanford University

[3] From Word Embeddings to Sentence Meanings [(slides)](https://simons.berkeley.edu/sites/default/files/docs/6449/christophermanning.pdf) [(video)](https://www.youtube.com/watch?v=nFCxTtBqF5U). Chris Manning, Stanford University

[4] RNNs and Language Models [(slides)](http://web.stanford.edu/class/cs224n/slides/cs224n-2020-lecture06-rnnlm.pdf) [(video)](https://youtu.be/iWea12EAu6U). CS224n: NLP with Deep Learning (2019), Stanford University

[5] Machine Translation, Seq2Seq and Attention [(slides)](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture08-nmt.pdf) [(video)](https://youtu.be/XXtpJxZBa2c). CS224n: NLP with Deep Learning (2019), Stanford University

**NLP Books**

[6] Speech and Language Processing ([3rd ed. draft](https://web.stanford.edu/~jurafsky/slp3/)). Dan Jurafsky, James H. Martin

[7] Neural Network Methods for Natural Language Processing ([2017](http://u.cs.biu.ac.il/~yogo/nnlp.pdf)). Yoav Goldberg

[8] Deep Learning ([online Version](https://www.deeplearningbook.org/)). Ian Goodfellow, Yoshua Bengio, AAron Courville

**Papers**

[9] Efficient Estimation of Word Representations in Vector Space (word2vec paper), [Mikolov 2013](https://arxiv.org/pdf/1301.3781.pdf)

[10] Sequence to Sequence Learning with Neural Networks (NMT with RNNs paper), [Sutskever 2014](https://arxiv.org/pdf/1409.3215.pdf)

[11] Neural Machine Translation by Jointly Learning Align and Translate (Seq2Seq with Attention paper), [Bahdanau 2016](https://arxiv.org/pdf/1409.0473.pdf)
