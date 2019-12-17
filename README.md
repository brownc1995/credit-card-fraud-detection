# credit-card-fraud-detection
[![Build Status](https://travis-ci.com/brownc1995/credit-card-fraud-detection.svg?branch=master)](https://travis-ci.com/brownc1995/credit-card-fraud-detection)
[![codecov](https://codecov.io/gh/brownc1995/credit-card-fraud-detection/branch/master/graph/badge.svg)](https://codecov.io/gh/brownc1995/credit-card-fraud-detection)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A range of models used for detecting fraudulent credit card 
transactions.

## Installation
You can install the project dependencies by running

```shell script
pip install -r requirements.txt
```


## Experiments
You can run the following line to start some experiments:
```shell script
python -m ccfd --epochs=25
```
You can view the fitting history in [logs](doc/logs) and display them 
using TensorBoard. 

The above command will run the
- simple logistic regression;
- random forest classifiers; 
- vanilla neural network;
- class-weighted neural network, and;
- oversampled neural network

experiments and save the history of the neural networks in 
[logs](doc/logs). 

The class-weighted neural network maps 
class indices (integers) to a weight (float) value, used for 
weighting the loss function (during training only). This can be 
useful to tell the model to "pay more attention" to samples from 
an under-represented class (such as the fraudulent transactions in
our data).

One can also oversample from the positive class such that we train
on data that has an equal distribution of positive and negative
examples. The network is then able to learn better how to distinguish
positive from negative examples.


## Testing
You can run tests for the repository using
```shell script
pytest --cov=.
```


## TensorBoard
Run the following to startup [TensorBoard](https://www.tensorflow.org/tensorboard) and view how your 
experiments perform:
```shell script
tensorboard --logdir doc/logs
``` 

## Results
Below are some results from each of the models stated above. The 
three neural networks were each trained for 25 epochs. The 
integers in parentheses denote the number of trees in each
forest.

| Model                  | accuracy | auc     | f-score |
| ---                    | ---      | ---     | ---     |
| Logistic Regression    | 0.99917  | 0.83840 | 0.73446 |
| Random Forest (1)      | 0.99912  | 0.85917 | 0.73404 |
| Random Forest (10)     | 0.99954  | 0.89058 | 0.85227 |
| Random Forest (100)    | 0.99954  | 0.90618 | 0.85714 |
| vanilla network        | 0.99853  | 0.90723 | 0.45455 |
| class-weighted network | 0.99837  | 0.90723 | 0.30075 |
| oversampled network    | 0.99851  | 0.90723 | 0.42953 |


## Context
It is important that credit card companies are able to recognise 
fraudulent credit card transactions so that customers are not 
charged for items that they did not purchase.


## Content
The [dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) contains transactions made by credit cards in 
September 2013 by European cardholders. This dataset presents 
transactions that occurred over two days, where we have 492 frauds 
out of 284,807 transactions. The dataset is highly imbalanced, 
the positive class (frauds) accounting for only 0.172% of all transactions.

The data contains only numerical input variables which are the result
of a PCA transformation. Unfortunately, due to confidentiality 
issues, Kaggle were not able to provide the original features. Features `V1`, `V2`,..., 
`V28` are the principal components obtained with PCA. The only 
features which have not been transformed with PCA are `Time` 
and `Amount`. Feature `Time` contains the seconds elapsed between 
each transaction and the first transaction in the dataset. The 
feature `Amount` is the transaction amount. This feature can be 
used for example-dependant cost-senstive learning. Feature `Class`
is the response variable and it takes value `1` in case of __fraud__ 
and `0` otherwise.

The TensorFlow [tutorial](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)
on imbalanced data heavily influenced the analysis performed in this project.


## Inspiration
Identify fraudulent credit card transactions.

Given the class imbalance ratio, [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) 
recommends measuring the 
accuracy using the Area Under the Precision-Recall Curve (AUPRC). 
Confusion matrix accuracy is not meaningful for unbalanced 
classification.


## Further work
The following are to-dos/areas for further investigation:
- try and use `tf.data.Dataset` everywhere and never `pd.DataFrame`
- build `tf.Extended` pipeline (see [here](https://www.tensorflow.org/tfx))


## References
- [Classification on imbalanced data; TensorFlow](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)
- [Credit Card Fraud Detection; Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
