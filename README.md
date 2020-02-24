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

In this section, we train 4 different RNNs for the taks of sentiment analysis

I. RNN with 1 bidirectional LSTM layer and word embeding layer trained from scratch

II. RNN with 1 bidirectional LSTM layer and a fine-tuned pretrained word embedding layer (GloVe)

III. RNN with 1 bidirectional LSTM layer and a frozen word embedding layer (Glove)

### 4a) Neural Machine Translation with Sequence-to-Sequence (seq2seq) Models



## References

[1] Introduction and Word Vectors [(slides)](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture01-wordvecs1.pdf) [(video)](https://youtu.be/8rXD5-xhemo). CS224n: NLP with Deep Learning (2019), Stanford University

[2] Word Vectors 2 and Word Senses [(slides)](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture02-wordvecs2.pdf) [(video)](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/cs224n-2019-notes02-wordvecs2.pdf). CS224n: NLP with Deep Learning (2019), Stanford University

[3] Representations for Language: From Word Embeddings to Sentence Meanings [(slides)](https://simons.berkeley.edu/sites/default/files/docs/6449/christophermanning.pdf) [(video)](https://www.youtube.com/watch?v=nFCxTtBqF5U). Chris Manning, Stanford University

[4] Recurrent Neural Networks and Language Models [(slides)](http://web.stanford.edu/class/cs224n/slides/cs224n-2020-lecture06-rnnlm.pdf) [(video)](https://youtu.be/iWea12EAu6U). CS224n: NLP with Deep Learning (2019), Stanford University

[5] Machine Translation, Seq2Seq and Attention [(slides)](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture08-nmt.pdf) [(video)](https://youtu.be/XXtpJxZBa2c). CS224n: NLP with Deep Learning (2019), Stanford University

