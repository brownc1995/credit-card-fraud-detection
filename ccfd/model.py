import logging
import os
from datetime import datetime
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from ccfd import *

logger = logging.getLogger(__name__)

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

BUFFER_SIZE = 100000


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

    logging.info(f'Weight for class 0: {weight_for_0:.2f}')
    logging.info(f'Weight for class 1: {weight_for_1:.2f}')

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
    cm = confusion_matrix(labels, predictions > 0.5)

    logging.info('==================================================================')

    for name, value in zip(model.metrics_names, results):
        logging.info(f'{name}: {value}')

    logging.info(f'Legitimate Transactions Detected (True Negatives): {cm[0][0]}')
    logging.info(f'Legitimate Transactions Incorrectly Detected (False Positives): {cm[0][1]}')
    logging.info(f'Fraudulent Transactions Missed (False Negatives): {cm[1][0]}')
    logging.info(f'Fraudulent Transactions Detected (True Positives): {cm[1][1]}')
    logging.info(f'Total Fraudulent Transactions: {np.sum(cm[1])}')

    logging.info('==================================================================')

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
        logging.info('Training using class-weighted data')
    elif resampled:
        logging.info('Training using resampled data')
    else:
        logging.info('Training using vanilla data')

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


def _make_dataset_pos_neg_helper(
        data: pd.DataFrame,
        target: pd.DataFrame
) -> tf.data.Dataset:
    """
    Helper for making positive/negative transaction datasets
    :param data: pd.DataFrame, features data
    :param target: pd.DataFrame, target data
    :return: tf.data.Dataset, resulting combined dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((data.values, target.values))
    dataset = dataset.shuffle(BUFFER_SIZE).repeat()

    return dataset


def make_datasets_pos_neg(
        pos_data: pd.DataFrame,
        pos_target: pd.DataFrame,
        neg_data: pd.DataFrame,
        neg_target: pd.DataFrame
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Dataset of poitive/negative transactions
    :param pos_data: pd.DataFrame, positive transaction features
    :param pos_target: pd.DataFrame, positive transaction target
    :param neg_data: pd.DataFrame, negative transaction features
    :param neg_target: pd.DataFrame, negative transaction target
    :return: Tuple[tf.data.Dataset, tf.data.Dataset], dataset of poitive/negative transactions
    """
    pos_dataset = _make_dataset_pos_neg_helper(pos_data, pos_target)
    neg_dataset = _make_dataset_pos_neg_helper(neg_data, neg_target)

    return pos_dataset, neg_dataset


def resample_dataset(
        pos_dataset: tf.data.Dataset,
        neg_dataset: tf.data.Dataset
) -> tf.data.Dataset:
    """
    Resampled dataset of the positive/negative datasets so that they have equal weighting in the resulting dataset
    :param pos_dataset: tf.data.Dataset, positive transaction dataset
    :param neg_dataset: tf.data.Dataset, negative transaction dataset
    :return: tf.data.Dataset, resampled dataset
    """
    resampled_dataset = tf.data.experimental.sample_from_datasets([pos_dataset, neg_dataset], weights=[0.5, 0.5])
    resampled_dataset = resampled_dataset.batch(BATCH_SIZE).prefetch(2)

    return resampled_dataset


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


def _make_dataset_helper(
        data: pd.DataFrame,
        target: pd.DataFrame
) -> tf.data.Dataset:
    """
    Helper for making datasets
    :param data: pd.DataFrame, features dataframe
    :param target: pd.DataFrame, target dataframe
    :return: tf.data.Dataset, dataset of combined dataframes
    """
    dataset = tf.data.Dataset.from_tensor_slices((data.values, target.values))
    dataset = dataset.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

    return dataset


def make_all_datasets(
        train_data: pd.DataFrame,
        train_target: pd.DataFrame,
        val_data: pd.DataFrame,
        val_target: pd.DataFrame,
        test_data: pd.DataFrame,
        test_target: pd.DataFrame
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Make datasets for each of our training, validation, and test datasets.
    :param train_data: pd.DataFrame, self-explanatory
    :param train_target: pd.DataFrame, self-explanatory
    :param val_data: pd.DataFrame, self-explanatory
    :param val_target: pd.DataFrame, self-explanatory
    :param test_data: pd.DataFrame, self-explanatory
    :param test_target: pd.DataFrame, self-explanatory
    :return: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset], datasets for each of our training, validation,
    and test datasets.
    """
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data.values, train_target.values))
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_data.values, val_target.values))
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(2)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_data.values, test_target.values))
    test_dataset = test_dataset.batch(BATCH_SIZE)

    return train_dataset, val_dataset, test_dataset
