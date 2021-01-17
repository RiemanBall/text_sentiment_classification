# Text Sentiment Classification

## Introduction
This repository contains two notebooks for Twitter tweet sentiment classification -- one using Bi-LSTM and the other one using pre-trained BERT.

The tweet dataset contains:
- labelled training data: 200k
- unlabeled training data: 1.1M
- testing data: 200k


## Models
#### Bi-LSTM:
To use the Bi-LSTM, we utilize the pre-trained Word2Vec model in Gensim to embed each sentence. In addition, Kerastuner is used to find the best hyperparameters.


#### BERT:
Here, we use the pre-trained BERT model from Tensorflow Hub. 


## Training
To make use of the unlabeled training dataset, we use semi-supervised learning to increase our training dataset.