"""Contains util functions"""
from typing import Tuple, List

import numpy as np
import pandas as pd
import csv

def load_data(filename: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """This function takes in the filepath of the data and outputs the tuple of 
    (X, Y, attributes_name). This reader assumes the label Y is always positioned 
    at the last column

    Parameters
    ----------
    filename: type `str`
        The filepath to the dataset

    Returns: a tuple (X, Y, attributes_name) where
    X: type `np.ndarray`, shape (N, M)
        Numpy arrays with N rows, M columns containing the attribute values for N training instances
    Y: type `np.ndarray`, shape (N, )
        Numpy arrays with N rows containing the true labels for N training instances
    split_attribute: type `int`
        The index of the attribute to split the dataset on
    """
    X: List[str] = []
    Y: List[str] = []
    attribute_names: List[str] = []
    with open(filename, 'r') as f: 
        reader = csv.reader(f)

        # get feature list, get rid of label header
        attribute_names = next(reader)[:-1] 

        # get X, Y values and convert them to numpy arrays
        for row in reader: 
            X.append(row[:-1])
            Y.append(row[-1])
        X = np.array(X)
        Y = np.array(Y)

    return (X, Y, attribute_names)


def error_rate(Y: np.ndarray, Y_pred: np.ndarray) -> float:
    
    incorrect: int = 0
    total: int = Y.shape[0]
    for y, y_pred in zip(Y, Y_pred): 
        if y != y_pred: incorrect += 1
    
    return incorrect / total