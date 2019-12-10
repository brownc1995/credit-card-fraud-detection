import logging
import os
from datetime import datetime
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, recall_score, precision_score, accuracy_score

from ccfd import CLASS, STEPS_PER_EPOCH, BATCH_SIZE

logger = logging.getLogger(__name__)

LOG_DIR = 'doc/logs'

RANDOM_FOREST_NUM_TREES = 10

METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
]


def _calc_class_sizes(
        df: pd.DataFrame
) -> Tuple[int, int, int]:
    """
    Helper function to calculate the size of each class in a given dataframe
    :param df: pd.DataFrame, input dataframe
    :return: Tuple[int, int, int], the total, positive, and negative number of transactions
    """
    total = len(df)
    pos = df[CLASS].sum()
    neg = total - pos

    return total, pos, neg


def calc_initial_bias(
        df: pd.DataFrame
) -> float:
    """
    Calculate the initial bias of the output layer of our network
    :param df: pd.DataFrame, input dataframe
    :return: float, initial bias of output layer of network
    """
    _, pos, neg = _calc_class_sizes(df)

    input_bias = np.log(pos / neg)

    return input_bias


def set_class_weights(
        df: pd.DataFrame
) -> dict:
    """
    Weight classes appropriately to aid training
    :param df: pd.DataFrame
    :return: dict, class and its weight
    """
    total, pos, neg = _calc_class_sizes(df)

    weight_for_0 = (1 / neg) * (total / 2)
    weight_for_1 = (1 / pos) * (total / 2)

    class_weight = {
        0: weight_for_0,
        1: weight_for_1
    }

    logger.info(f'Weight for class 0: {weight_for_0:.2f}')
    logger.info(f'Weight for class 1: {weight_for_1:.2f}')

    return class_weight


def _layers_list() -> List[tf.keras.layers.Layer]:
    """
    Helper function to create list of layers for network
    :return: List[tf.keras.layers.Layer], list of layers
    """
    num_nodes_list = [30, 60, 120, 240, 480]
    num_nodes_list += num_nodes_list[::-1]
    layers_list = []
    for num_nodes in num_nodes_list:
        layers_list += [
            tf.keras.layers.Dense(num_nodes, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5)
        ]

    return layers_list


def build_model(
        input_shape: Union[int, Tuple[int,]],
        output_bias: float = None,
) -> tf.keras.Model:
    """
    Build neural network model
    :param input_shape: Union[int, Tuple[int,]], shape of input data
    :param output_bias: float, value of output biase
    :return: tf.keras.Model, neural network model
    """
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    input_shape = (input_shape,) if isinstance(input_shape, int) else input_shape

    layers = [tf.keras.layers.InputLayer(input_shape)]
    layers += _layers_list()
    layers += [tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)]

    model = tf.keras.Sequential(layers)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS
    )

    return model


def _log_fscore(
        metric_values_map: dict
) -> None:
    """
    Log f-score value
    :param metric_values_map: dict, map of metric name to value
    :return: None
    """
    precision = metric_values_map['precision']
    recall = metric_values_map['recall']
    fscore = 2 * precision * recall / (precision + recall)

    logger.info(f'f1_score: {fscore}')

    return None


def _log_confusion_matrix(
        labels: Union[pd.Series, np.ndarray],
        predictions: Union[pd.Series, np.ndarray]
) -> None:
    """
    Log confusion matrix values
    :param labels: Union[pd.Series, np.ndarray], series/array of labels
    :param predictions: Union[pd.Series, np.ndarray], series/array of predicted outputs
    :return: None
    """
    tn, fp, fn, tp = confusion_matrix(labels, predictions > 0.5).ravel()

    logger.info(f'Legitimate Transactions Detected (True Negatives): {tn}')
    logger.info(f'Legitimate Transactions Incorrectly Detected (False Positives): {fp}')
    logger.info(f'Fraudulent Transactions Missed (False Negatives): {fn}')
    logger.info(f'Fraudulent Transactions Detected (True Positives): {tp}')
    logger.info(f'Total Fraudulent Transactions: {fn + tp}')

    return None


def log_model_performance(
        model: tf.keras.Model,
        results: List[float],
        labels: Union[pd.Series, np.ndarray],
        predictions: Union[pd.Series, np.ndarray]
) -> None:
    """
    Log model performance
    :param model: tf.keras.Model, neural network model
    :param results: List[float], list of results of model performance
    :param labels: Union[pd.Series, np.ndarray], series/array of labels
    :param predictions: Union[pd.Series, np.ndarray], series/array of predicted outputs
    :return: None
    """
    metric_values_map = dict(zip(model.metrics_names, results))

    logger.info('==================================================================')

    for name, value in metric_values_map.items():
        logger.info(f'{name}: {value}')

    _log_fscore(metric_values_map)

    _log_confusion_matrix(labels, predictions)

    logger.info('==================================================================')

    return None


def _tensorboard_callback(
        class_weight: dict,
        resampled: bool
) -> tf.keras.callbacks.TensorBoard:
    """
    Helper function to create TensorBoard callback
    :param class_weight: dict, output of set_class_weights
    :param resampled: bool, has data been resampled?
    :return: tf.keras.callbacks.TensorBoard, TensorBoard callback
    """
    logdir = os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))

    if class_weight is not None:
        logdir += '_cw'
    elif resampled:
        logdir += '_rs'

    return tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)


def fit_model(
        model: tf.keras.Model,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        epochs: int,
        steps_per_epoch: int = STEPS_PER_EPOCH,
        class_weight: dict = None,
        resampled: bool = False
) -> tf.keras.callbacks.History:
    """
    Fit neural network model
    :param model: tf.keras.Model, neural network model
    :param train_dataset: tf.data.Dataset, dataset to train on
    :param val_dataset: tf.data.Dataset, dataset for validation
    :param epochs: int, number of epochs to fit on
    :param steps_per_epoch: int, number of steps to take in each epoch
    :param class_weight: dict, output of set_class_weights
    :param resampled: bool, has data been resampled?
    :return: tf.keras.callbacks.History, history of fitted model over epochs
    """
    assert not (resampled and class_weight is not None), 'Only one of resampled/class_weight should be True/not None.'

    if class_weight is not None:
        logger.info('Training using class-weighted data')
    elif resampled:
        logger.info('Training using resampled data')
    else:
        logger.info('Training using vanilla data')

    tensorboard_callback = _tensorboard_callback(class_weight, resampled)

    history = model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        callbacks=[tensorboard_callback],
        class_weight=class_weight
    )

    return history


def resample_steps_per_epoch(
        df: pd.DataFrame
) -> int:
    """
    Steps per epoch to take for resampled data. The definition of "epoch" in this case is less clear. Say it's the
    number of batches required to see each negative example once.
    :param df: pd.DataFrame, input dataframe
    :return: int, steps per epoch for resampled data
    """
    _, _, neg = _calc_class_sizes(df)

    return int(np.ceil(2 * neg / BATCH_SIZE))


def _calc_sklearn_metrics(
        test_target: pd.Series,
        test_pred: pd.Series
) -> dict:
    tn, fp, fn, tp = confusion_matrix(test_target, test_pred).ravel()

    metrics_d = {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'accuracy': accuracy_score(test_target, test_pred),
        'precision': precision_score(test_target, test_pred),
        'recall': recall_score(test_target, test_pred),
        'f1_score': f1_score(test_target, test_pred),
        'auc': roc_auc_score(test_target, test_pred)
    }

    return metrics_d


def simple_model(
        train_data: pd.DataFrame,
        train_target: pd.Series,
        test_data: pd.DataFrame,
        test_target: pd.Series,
        model_type: str,
        log: bool = True,
        num_trees: int = None
) -> dict:
    """
    Run simple logistic regression model and return metrics
    :param train_data: pd.DataFrame, training features
    :param train_target: pd.DataFrame, training target variable
    :param test_data: pd.DataFrame, test features
    :param test_target: pd.DataFrame, test target variable
    :param model_type: str,
    :param log: bool, to log or not to log metrics
    :param num_trees: int, number of trees to run random forest with
    :return: dict, dictionary of metrics
    """
    exp_model_type = ['logistic_regression', 'random_forest']
    assert model_type in exp_model_type, f'model_type must be in {exp_model_type}. It is currently {model_type}.'

    if model_type == 'logistic_regression':
        logger.info('Running simple logistic regression model')
        clf = LR(solver='lbfgs')
    else:
        n_estimators = num_trees if num_trees is not None else RANDOM_FOREST_NUM_TREES
        logger.info(f'Running random forest model with {n_estimators} tree(s).')
        clf = RFC(n_estimators=n_estimators)

    clf.fit(train_data, train_target)
    test_pred = clf.predict(test_data)
    metrics_d = _calc_sklearn_metrics(test_target, test_pred)

    if log:
        logger.info('==================================================================')

        for name, value in metrics_d.items():
            logger.info(f'{name}: {value}')

        _log_confusion_matrix(test_target, test_pred)

        logger.info('==================================================================')

    return metrics_d
