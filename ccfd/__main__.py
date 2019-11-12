import argparse
import logging
from typing import Optional, Tuple

import pandas as pd
import tensorflow as tf

from ccfd import STEPS_PER_EPOCH
from ccfd.data import get_data, scale_data, train_val_test_split, pos_neg_data
from ccfd.model import calc_initial_bias, set_class_weights, make_all_datasets, resample_steps_per_epoch, \
    resample_dataset, make_datasets_pos_neg, build_model, fit_model, log_model_performance

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS = [
    'vanilla',
    'class_weighted',
    'resampled'
]


def _prepare_resampled_data(
        train_data: pd.DataFrame,
        train_target: pd.DataFrame,
        ccfd_data: pd.DataFrame
) -> Tuple[tf.data.Dataset, int]:
    """
    Resample training dataset
    :param train_data: pd.DataFrame, training features
    :param train_target: pd.DataFrame, training target
    :param ccfd_data: pd.DataFrame, original dataframe
    :return: Tuple[tf.data.Dataset, int], resampled dataset and the number of steps we will take per epoch
    """
    pos_data, pos_target, neg_data, neg_target = pos_neg_data(train_data, train_target)
    pos_dataset, neg_dataset = make_datasets_pos_neg(pos_data, pos_target, neg_data, neg_target)
    resampled_dataset = resample_dataset(pos_dataset, neg_dataset)
    resampled_steps_per_epoch = resample_steps_per_epoch(ccfd_data)

    return resampled_dataset, resampled_steps_per_epoch


def setup(
        epochs: int
) -> tuple:
    """
    Setup for running neural network experiments
    :param epochs: int, number of epochs to run
    :return: tuple, inputs for function to run neural network
    """
    ccfd_data = get_data()

    train_data, train_target, val_data, val_target, test_data, test_target = train_val_test_split(ccfd_data)

    train_data, val_data, test_data = scale_data(train_data, val_data, test_data)

    train_dataset, val_dataset, test_dataset = make_all_datasets(
        train_data,
        train_target,
        val_data,
        val_target,
        test_data,
        test_target
    )

    logger.info('Setting up neural network')
    initial_bias = calc_initial_bias(ccfd_data)
    input_shape = train_data.shape[-1]

    class_weight = set_class_weights(ccfd_data)

    resampled_dataset, resampled_steps_per_epoch = _prepare_resampled_data(train_data, train_target, ccfd_data)

    nn_kwargs = {
        'initial_bias': initial_bias,
        'input_shape': input_shape,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'test_target': test_target,
        'epochs': epochs
    }

    return class_weight, resampled_dataset, resampled_steps_per_epoch, nn_kwargs


def run_network(
        initial_bias: float,
        input_shape: int,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset,
        test_target: pd.Series,
        epochs: int,
        model_type: str,
        class_weight: Optional[dict] = None,
        steps_per_epoch: int = STEPS_PER_EPOCH,
        resampled: bool = False
) -> None:
    """
    Run neural network experiment
    :param initial_bias: float, output layer's bias to reflect imbalance in data
    :param input_shape: int, shape of input data for network
    :param train_dataset: tf.data.Dataset, train dataset
    :param val_dataset: tf.data.Dataset, validation dataset
    :param test_dataset: tf.data.Dataset, test dataset
    :param test_target: pd.Series, test target
    :param epochs: int, number of epochs to run network over
    :param model_type: str, vanilla, class-weighted, or resampled?
    :param class_weight: Optional[dict], class-weightings
    :param steps_per_epoch: int, steps taken per epoch
    :param resampled: bool, resample data?
    :return: None
    """
    assert model_type in MODELS, f'model_type is currently {model_type}. Should be one of {MODELS}'

    logger.info(f'Creating {model_type} model')
    model = build_model(
        input_shape=input_shape,
        output_bias=initial_bias
    )

    logger.info(f'Fitting {model_type} model')
    fit_model(
        model,
        train_dataset,
        val_dataset,
        epochs=epochs,
        class_weight=class_weight,
        steps_per_epoch=steps_per_epoch,
        resampled=resampled
    )

    logger.info('Evaluating model performance')
    baseline_results = model.evaluate(
        test_dataset,
        verbose=0
    )
    test_predictions_baseline = model.predict(test_dataset)
    log_model_performance(model, baseline_results, test_target, test_predictions_baseline)

    return None


def main(
        epochs: int,
) -> None:
    """
    Main function for experiments
    :param epochs: int, number of epochs to run over
    :return: None
    """
    class_weight, resampled_dataset, resampled_steps_per_epoch, nn_kwargs = setup(epochs)

    run_network(model_type='vanilla', **nn_kwargs)

    run_network(model_type='class_weighted', class_weight=class_weight, **nn_kwargs)

    run_network(model_type='resampled', steps_per_epoch=resampled_steps_per_epoch, resampled=True, **nn_kwargs)

    logger.info('Experiments ran successfully')

    logger.info('Exiting')

    return None


def _arg_parser() -> argparse.Namespace:
    """
    Argument parser
    :return: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Main entry point for CCFD model')

    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        help='number of epochs to train models over',
        required=True
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = _arg_parser()
    main(epochs=args.epochs)
