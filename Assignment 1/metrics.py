from typing import Union
import pandas as pd
import numpy as np

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """ Function to calculate the accuracy """
    assert y_hat.size == y.size and y_hat.size > 0
    return (y_hat==y).sum()/len(y_hat)


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """ Function to calculate the precision """
    true_positives = ((y_hat == cls) & (y == cls)).sum()
    false_positives = ((y_hat == cls) & (y != cls)).sum()
    denominator = true_positives + false_positives
    if denominator == 0:
        return 0.0
    return true_positives / denominator


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """ Function to calculate the recall """
    true_positives = ((y_hat == cls) & (y == cls)).sum()
    false_negatives = ((y_hat != cls) & (y == cls)).sum()
    denominator = true_positives+false_negatives
    if denominator == 0:
        return 0.0
    return true_positives/denominator

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """ Function to calculate the root-mean-squared-error(rmse) """
    assert y_hat.size == y.size and y_hat.size > 0
    return np.sqrt(((y - y_hat)**2).mean())

def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """ Function to calculate the mean-absolute-error(mae) """
    assert y_hat.size == y.size and y_hat.size > 0
    return np.abs(y-y_hat).mean()