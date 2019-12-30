import logging
from typing import Tuple, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ccfd import CLASS, BATCH_SIZE

logger = logging.getLogger(__name__)

CCFD_DATA_PATH = 'doc/data/creditcard'

BUFFER_SIZE = 100000


def get_data(
        load_original: bool = False
) -> pd.DataFrame:
    """
    Load transaction data. If load_original=True, note that column names have been made lower case and
    that log_amount column has been added to the dataframe.
    :param load_original: bool, load from csv?
    :return: pd.DataFrame, transaction data
    """
    logger.info('Grabbing data')
    if load_original:
        data = pd.read_csv(f'{CCFD_DATA_PATH}.csv')
        data.columns = [c.lower() for c in data.columns]
        data.drop(['time'], inplace=True, axis=1)
    else:
        data = pd.read_pickle(f'{CCFD_DATA_PATH}.pkl')  # pkl file is result of above formatting of csv

    logger.info(f'We have {len(data)} records of which {data["class"].sum()} were fraudulent.')
    logger.info(f'{data["class"].sum() / len(data) * 100:.3f}% of records are fraudulent.')

    return data


def _feature_target_split(
        df_seq: Sequence[pd.DataFrame]
) -> Tuple:
    """
    Split dataframe into feature and target dataframes
    :param df_seq: Sequence[pd.DataFrame], sequence of dataframes to split
    :return: Tuple, resulting split dataframes
    """
    data_target_list = []
    for df in df_seq:
        data_target_list += [df.drop([CLASS], axis=1), df[CLASS]]
    return tuple(data_target_list)


def train_val_test_split(
        df: pd.DataFrame,
        test_size: float = 0.2
) -> Tuple:
    """
    Split data into train, val and test data. Note that val uses test_size also but is split on the train data i.e.
    val is test_size**2 of the original data
    :param df: pd.DataFrame, data to split on
    :param test_size: float, proportion of training set set aside for test and val data.
    :return: Tuple, train, test and val features and targets
    """
    logger.info('Splitting data into training, validation, and test sets')

    train_df, test_df = train_test_split(df, test_size=test_size)
    train_df, val_df = train_test_split(train_df, test_size=test_size)

    feature_target = _feature_target_split([train_df, val_df, test_df])

    train_data, train_target, val_data, val_target, test_data, test_target = feature_target

    return train_data, train_target, val_data, val_target, test_data, test_target


def log_shapes(
        train_data: pd.DataFrame,
        train_target: pd.DataFrame,
        val_data: pd.DataFrame,
        val_target: pd.DataFrame,
        test_data: pd.DataFrame,
        test_target: pd.DataFrame,
) -> None:  # pragma: no cover
    """
    Log shapes of dataframes
    :param train_data: pd.DataFrame, self-explanatory
    :param train_target: pd.DataFrame, self-explanatory
    :param val_data: pd.DataFrame, self-explanatory
    :param val_target: pd.DataFrame, self-explanatory
    :param test_data: pd.DataFrame, self-explanatory
    :param test_target: pd.DataFrame, self-explanatory
    :return: None
    """
    logger.info(f'Training data shape: {train_data.shape}')
    logger.info(f'Validation data shape: {val_data.shape}')
    logger.info(f'Test data shape: {test_data.shape}')
    logger.info(f'Training labels shape: {train_target.shape}')
    logger.info(f'Validation labels shape: {val_target.shape}')
    logger.info(f'Test labels shape: {test_target.shape}')

    return None


def scale_data(
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame
) -> Tuple[pd.DataFrame, ...]:
    """
    Normalise data to have mean=0, var=1
    :param train_data: pd.DataFrame, self-explanatory
    :param val_data: pd.DataFrame, self-explanatory
    :param test_data: pd.DataFrame, self-explanatory
    :return: Tuple[pd.DataFrame, ...], scaled dataframes
    """
    logger.info('Normalising input data')

    cols = train_data.columns

    scaler = StandardScaler()
    train_data = pd.DataFrame(scaler.fit_transform(train_data), columns=cols)
    val_data = pd.DataFrame(scaler.transform(val_data), columns=cols)
    test_data = pd.DataFrame(scaler.transform(test_data), columns=cols)

    return train_data, val_data, test_data


# TODO: write test
def pos_neg_data(
        train_data: pd.DataFrame,
        train_target: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split training data into those rows with positive/negative transactions
    :param train_data: pd.DataFrame, self-explanatory
    :param train_target: pd.DataFrame, self-explanatory
    :return: Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], data split into those rows with positive/negative
    transactions.
    """
    bool_train_target = np.array(train_target) != 0

    pos_data = train_data[bool_train_target]
    neg_data = train_data[~bool_train_target]

    pos_target = train_target[bool_train_target]
    neg_target = train_target[~bool_train_target]

    return pos_data, pos_target, neg_data, neg_target


# TODO: write test
def resample_dataframe(
        pos_data: pd.DataFrame,
        pos_target: pd.DataFrame,
        neg_data: pd.DataFrame,
        neg_target: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Resample dataframe and return as dataframe
    :param pos_data: pd.DataFrame, positive features
    :param pos_target: pd.DataFrame, positive target
    :param neg_data: pd.DataFrame, negative features
    :param neg_target: pd.DataFrame, negative target
    :return: Tuple[pd.DataFrame, pd.Series], resampled training features and target
    """
    ids = np.arange(len(pos_data))
    choices = np.random.choice(ids, len(neg_data))

    res_pos_data = pos_data[choices]
    res_pos_target = pos_target[choices]

    resampled_data = np.concatenate([res_pos_data, neg_data], axis=0)
    resampled_target = np.concatenate([res_pos_target, neg_target], axis=0)

    order = np.arange(len(resampled_target))
    np.random.shuffle(order)
    resampled_data = resampled_data[order]
    resampled_target = resampled_target[order]

    return resampled_data, resampled_target


# TODO: write test
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
    logger.info('Resampling data')
    resampled_dataset = tf.data.experimental.sample_from_datasets([pos_dataset, neg_dataset], weights=[0.5, 0.5])
    resampled_dataset = resampled_dataset.batch(BATCH_SIZE).prefetch(2)

    return resampled_dataset


def _make_dataset_helper(
        data: pd.DataFrame,
        target: pd.DataFrame
) -> tf.data.Dataset:  # pragma: no cover
    """
    Helper for making datasets
    :param data: pd.DataFrame, features dataframe
    :param target: pd.DataFrame, target dataframe
    :return: tf.data.Dataset, dataset of combined dataframes
    """
    dataset = tf.data.Dataset.from_tensor_slices((data.values, target.values))
    dataset = dataset.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

    return dataset


# TODO: write test
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
    logger.info('Transforming data from pd.DataFrames to tf.data.Datasets')

    train_dataset = tf.data.Dataset.from_tensor_slices((train_data.values, train_target.values))
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_data.values, val_target.values))
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(2)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_data.values, test_target.values))
    test_dataset = test_dataset.batch(BATCH_SIZE)

    return train_dataset, val_dataset, test_dataset


def _make_dataset_pos_neg_helper(
        data: pd.DataFrame,
        target: pd.Series
) -> tf.data.Dataset:  # pragma: no cover
    """
    Helper for making positive/negative transaction datasets
    :param data: pd.DataFrame, features data
    :param target: pd.DataFrame, target data
    :return: tf.data.Dataset, resulting combined dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((data.values, target.values))
    dataset = dataset.shuffle(BUFFER_SIZE).repeat()

    return dataset


# TODO: write test
def make_datasets_pos_neg(
        pos_data: pd.DataFrame,
        pos_target: pd.Series,
        neg_data: pd.DataFrame,
        neg_target: pd.Series
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
