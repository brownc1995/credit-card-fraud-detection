from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

FIGSIZE = (8, 7)


def plot_pos_neg(
        train_data: pd.DataFrame,
        train_target: pd.DataFrame,
        col1: str = 'v5',
        col2: str = 'v6'
) -> None:
    """
    Make hexbin plot for training transaction data
    :param train_data: pd.DataFrame, features dataframe
    :param train_target: pd.DataFrame, target dataframe
    :param col1: str, name of first column for hexbin plot
    :param col2: str, name of second column for hexbin plot
    :return: None
    """
    pos_df = pd.DataFrame(train_data[train_target.values == 1], columns=train_data.columns)
    neg_df = pd.DataFrame(train_data[train_target.values == 0], columns=train_data.columns)

    sns.jointplot(pos_df[col1], pos_df[col2], kind='hex', xlim=(-5, 5), ylim=(-5, 5))
    plt.suptitle('Positive distribution')

    sns.jointplot(neg_df[col1], neg_df[col2], kind='hex', xlim=(-5, 5), ylim=(-5, 5))
    _ = plt.suptitle('Negative distribution')

    return None


def plot_roc(
        name: str,
        labels: Union[pd.DataFrame, np.ndarray],
        predictions: Union[pd.DataFrame, np.ndarray],
        **kwargs
) -> None:
    """
    Helper function for plotting roc curves
    :param name: str, label of curve
    :param labels: Union[pd.DataFrame, np.ndarray], target values
    :param predictions: Union[pd.DataFrame, np.ndarray], predicted values
    :return: None
    """
    fp, tp, _ = roc_curve(labels, predictions)

    plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5, 20])
    plt.ylim([80, 100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

    return None


def plot_roc_all(
        train_target: Union[pd.Series, np.ndarray],
        test_target: Union[pd.Series, np.ndarray],
        train_predictions_baseline: Union[pd.Series, np.ndarray],
        test_predictions_baseline: Union[pd.Series, np.ndarray],
        train_predictions_cw: Union[pd.Series, np.ndarray],
        test_predictions_cw: Union[pd.Series, np.ndarray],
        train_predictions_rs: Union[pd.Series, np.ndarray],
        test_predictions_rs: Union[pd.Series, np.ndarray]
) -> None:
    """
    Plot ROC curves for data of interest
    :param train_target: Union[pd.Series, np.ndarray], self-explanatory
    :param test_target: Union[pd.Series, np.ndarray], self-explanatory
    :param train_predictions_baseline: Union[pd.Series, np.ndarray], self-explanatory
    :param test_predictions_baseline: Union[pd.Series, np.ndarray], self-explanatory
    :param train_predictions_cw: Union[pd.Series, np.ndarray], self-explanatory
    :param test_predictions_cw: Union[pd.Series, np.ndarray], self-explanatory
    :param train_predictions_rs: Union[pd.Series, np.ndarray], self-explanatory
    :param test_predictions_rs: Union[pd.Series, np.ndarray], self-explanatory
    :return: None
    """
    max_row = len(train_target)

    plt.figure(figsize=FIGSIZE)
    plot_roc('Train Baseline', train_target, train_predictions_baseline[:max_row], color=colors[0])
    plot_roc('Test Baseline', test_target, test_predictions_baseline, color=colors[0], linestyle='--')

    plot_roc('Train Weighted', train_target, train_predictions_cw[:max_row], color=colors[1])
    plot_roc('Test Weighted', test_target, test_predictions_cw, color=colors[1], linestyle='--')

    plot_roc('Train Resampled', train_target, train_predictions_rs[:max_row], color=colors[2])
    plot_roc('Test Resampled', test_target, test_predictions_rs, color=colors[2], linestyle='--')
    plt.legend(loc='lower right')

    plt.xlim((0, 100))
    plt.ylim((0, 100))

    plt.show()

    return None
