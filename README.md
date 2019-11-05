# credit-card-fraud-detection
[![Build Status](https://travis-ci.com/brownc1995/credit-card-fraud-detection.svg?branch=master)](https://travis-ci.com/brownc1995/credit-card-fraud-detection)
[![codecov](https://codecov.io/gh/brownc1995/credit-card-fraud-detection/branch/master/graph/badge.svg)](https://codecov.io/gh/brownc1995/credit-card-fraud-detection)

A range of models used for detecting fraudulent credit card 
transactions.

## Installation

You can install the project dependencies by running

```shell script
pip install -r requirements.txt
```

## Context
It is important that credit card companies are able to recognize 
fraudulent credit card transactions so that customers are not 
charged for items that they did not purchase.

## Content
The [dataset](sample/data) contains transactions made by credit cards in 
September 2013 by European cardholders. This dataset presents 
transactions that occurred in two days, where we have 492 frauds 
out of 284,807 transactions. The dataset is highly unbalanced, 
the positive class (frauds) account for 0.172% of all transactions.

The data contains only numerical input variables which are the result
of a PCA transformation. Unfortunately, due to confidentiality 
issues, we cannot provide the original features and more 
background information about the data. Features `V1`, `V2`,..., 
`V28` are the principal components obtained with PCA, the only 
features which have not been transformed with PCA are `Time` 
and `Amount`. Feature `Time` contains the seconds elapsed between 
each transaction and the first transaction in the dataset. The 
feature `Amount` is the transaction amount. This feature can be 
used for example-dependant cost-senstive learning. Feature 'Class'
is the response variable and it takes value `1` in case of __fraud__ 
and `0` otherwise.

## Inspiration
Identify fraudulent credit card transactions.

Given the class imbalance ratio, [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) 
recommends measuring the 
accuracy using the Area Under the Precision-Recall Curve (AUPRC). 
Confusion matrix accuracy is not meaningful for unbalanced 
classification.

## TensorBoard
Run the following to startup [TensorBoard](https://www.tensorflow.org/tensorboard) and view how your 
experimnts perform:
```shell script
tensorboard --logdir doc/logs
``` 

## To Do
Still to do:
- Add experiments for user to run
