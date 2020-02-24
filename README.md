Introduction to NLP with Deep Learning
======================================

### 1) Introduction to NLP and Word Embeddings

### 2a) Natural Language Processing and Neural Networks

In this notebook, we build and train a shallow neural network for the sentiment analysis of the IMDB Reviews Dataset. Since regular neural networks are not designed for processing sequences of vaiable length, we use the Gloval Average Pooling to convert sequence of vectors into a vector of the sequence's average. In this manner, we turn every IMDB review into a vector that represents.. This average vector is then fed to a neural network with three layers: input layer, hidden layer and output layer.

The last layer contains only one neuron, and it represents the probability for a particular IMDB review to be positive. This last layer uses sigmoid as the activation function.

### 2b) Pretrained Sentence Embeddings and Neural Networks

In the previous notebook, we trained an embedding layer from scratch using a subwords representation of around 8,000 tokens. Now we will be working with pretrained sentence embeddings that have been trained on the Google News dataset (200 billion words).

Each IMDB review is fed to the pretrained sentence embedding layer and subsequently converted into a vector. This vector is the processed by a shallow neural network of 3 layers: input layer, hidden layer and output layer.

### 3a) Natural Language Processing and Recurrent Neural Networks

We now introduce the Recurrent Neural Networks that are capable of processing sequences of any length

### 3b) Pretrained Word Embedding and Recurrent Neural Networks



### 4a) Neural Machine Translation with Sequence-to-Sequence (seq2seq) Models
