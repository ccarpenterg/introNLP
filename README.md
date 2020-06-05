Introduction to NLP with Deep Learning
======================================

This repository is a collection of notebooks that serves as an introduction to Natural Language Processing, and it's loosely based on the [Natural Language Processing with Deep Learning (CS224n)](http://web.stanford.edu/class/cs224n/) course taught at Stanford University by [Chris Manning](https://nlp.stanford.edu/~manning/). The models are trained using [Tensorflow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/); the diagrams are drawn with [Draw.io](https://www.draw.io/) and the Math symbols are written with LaTeX.

This introduction assumes some background on engineering, specifically on Calculus, Linear Algebra, Probability and Statistics, and some experience programming Python.

### 1) Introduction to NLP and Word Embeddings

The first notebook introduces some of the fundamental concepts of Natural Language Processing with an emphasis on the aspects that are relevant to Deep Learning. It introduces the concepts of language modeling, N-grams, vector semantics, word embeddings, etc.

<img src="https://user-images.githubusercontent.com/114733/83787424-c546a000-a661-11ea-8f87-b334612f800e.jpg" title="embeddings" />

Notebook:
([Jupyter nbviewer](https://nbviewer.jupyter.org/github/ccarpenterg/introNLP/blob/master/01a_intro_NLP_and_word_embeddings.ipynb))
([github](https://github.com/ccarpenterg/introNLP/blob/master/01a_intro_NLP_and_word_embeddings.ipynb))

### 2a) Natural Language Processing and Neural Networks

In this notebook, we build and train a shallow neural network for the sentiment analysis of the IMDB Reviews Dataset. Since regular neural networks are not designed for processing sequences of variable length, we use the Gloval Average Pooling to convert sequence of word embedding vectors into one vector. In this manner, we turn every IMDB review into a vector that represents an embedding for the entire sequence. This vector is then fed to a neural network with three layers: input layer, hidden layer and output layer.

The last layer contains only one neuron, and it represents the probability for a particular IMDB review to be positive. This last layer uses sigmoid as the activation function. Our model is trained as an end-to-end neural network, meaning that the network learns both the word embeddings and the layers' weights at the same time.

Notebook:
([Jupyter nbviewer](https://nbviewer.jupyter.org/github/ccarpenterg/introNLP/blob/master/02a_NLP_and_neural_networks.ipynb))
([github](https://github.com/ccarpenterg/introNLP/blob/master/02a_NLP_and_neural_networks.ipynb))

### 2b) Pretrained Sentence Embeddings and Neural Networks

In the previous notebook, we trained an embedding layer from scratch using a subwords representation of around 8,000 tokens. Now we will be working with pretrained sentence embeddings that have been trained on the Google News dataset (200 billion words).

Each IMDB review is fed to the pretrained sentence embedding layer and subsequently converted into a vector. This vector is the processed by a shallow neural network of 3 layers: input layer, hidden layer and output layer. We use two different pretrained sentence embeddings and try two different approaches for training our neural network. In one case we freeze our pretrained sentence embeddings' weights and in the other, we fine-tune them.

Notebook:
([Jupyter nbviewer](https://nbviewer.jupyter.org/github/ccarpenterg/introNLP/blob/master/02b_NLP_and_neural_networks.ipynb))
([github](https://github.com/ccarpenterg/introNLP/blob/master/02b_NLP_and_neural_networks.ipynb))

### 3a) Natural Language Processing and Recurrent Neural Networks

In contrast with the feed-forward neural networks seen in the previous section, Recurrent Neural Networks (or RNNs) have the ability to process sequences of variable length and learn representations that capture the relationships between the sequence's elements.

<img src="https://user-images.githubusercontent.com/114733/78505113-3c81b580-773f-11ea-8a97-088fd306bdc4.jpg" title="RNN" />

Notebook:
([Jupyter nbviewer](https://nbviewer.jupyter.org/github/ccarpenterg/introNLP/blob/master/03a_NLP_and_recurrent_neural_networks.ipynb))
([github](https://github.com/ccarpenterg/introNLP/blob/master/03a_NLP_and_recurrent_neural_networks.ipynb))

### 3b) Pretrained Word Embedding and Recurrent Neural Networks

RNN are very flexible models and it is possible to use pretrained word embeddings instead of training a word embedding layer from scratch. This notebook presents three different scenarios: (i) training an embedding layer from zero, (ii) fine-tuning a pretrained embedding such as GloVe or word2vec, and (iii) frozing the layer with the pretrained embeddings.

Notebook:
([Jupyter nbviewer](https://nbviewer.jupyter.org/github/ccarpenterg/introNLP/blob/master/03b_NLP_and_recurrent_neural_networks.ipynb))
([github](https://github.com/ccarpenterg/introNLP/blob/master/03b_NLP_and_recurrent_neural_networks.ipynb))

### 4a) Neural Machine Translation with a Sequence-to-Sequence Model

Neural Machine Translation is the task of automatically translating text from a source language to a target language with neural networks. The Sequence to Sequence model is presented in this notebook, and a encoder-decoder model based on two RNNs, is trained on a parallel corpus of Spanish-English sentences.

<img src="https://user-images.githubusercontent.com/114733/79812038-af954980-8345-11ea-8f96-8a527fe3e029.jpg" title="seq2seq" />

Notebook:
([Jupyter nbviewer](https://nbviewer.jupyter.org/github/ccarpenterg/introNLP/blob/master/04a_NLP_and_sequence_to_sequence_RNNs.ipynb))
([github](https://github.com/ccarpenterg/introNLP/blob/master/04a_NLP_and_sequence_to_sequence_RNNs.ipynb))

### 4b) Neural Machine Translation with a Seq2Seq and Attention Model

The model presented in the previous notebook is part of a group of encoder-decoder models that encode a source sentence into a vector (the encoder RNN's hidden state) of fixed length from which the model generates a target sentence (the translation). Based on the premise that the use of this vector is a bottleneck, Bahdanau (2016) introduced the Sequence-to-Sequence with Attention model which automatically search for the elements in the source sentence that are relevant to predicting the next target sentence word, without explicitly representing these relationships.

The same parallel corpus of Spanish-English sentences is used to train a sequence to sequence model with attention model.

<img src="https://user-images.githubusercontent.com/114733/80148688-6b8f8800-8583-11ea-84f3-fb1d464bcebf.jpg" title="attention" />

Notebook: ([Jupyter nbviewer](https://nbviewer.jupyter.org/github/ccarpenterg/introNLP/blob/master/04b_NLP_and_sequence_to_sequence_RNNs.ipynb)) ([github](https://github.com/ccarpenterg/introNLP/blob/master/04b_NLP_and_sequence_to_sequence_RNNs.ipynb))

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

[8] Deep Learning ([online version](https://www.deeplearningbook.org/)). Ian Goodfellow, Yoshua Bengio, AAron Courville

**Papers**

[9] Recent Trends in Deep Learning Based Natural Language Processing, [Young 2018](https://arxiv.org/pdf/1708.02709.pdf)

[10] Efficient Estimation of Word Representations in Vector Space (word2vec paper), [Mikolov 2013](https://arxiv.org/pdf/1301.3781.pdf)

[11] Sequence to Sequence Learning with Neural Networks (NMT with RNNs paper), [Sutskever 2014](https://arxiv.org/pdf/1409.3215.pdf)

[12] Neural Machine Translation by Jointly Learning Align and Translate (Seq2Seq with Attention paper), [Bahdanau 2016](https://arxiv.org/pdf/1409.0473.pdf)
