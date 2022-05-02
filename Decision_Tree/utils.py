"""Contains util functions"""
from typing import Tuple, List

import numpy as np
import pandas as pd
import csv


def majority_vote(X: np.ndarray, Y: np.ndarray) -> str:
    """This function computes the output label of the given dataset, following the 
    majority vote algorithm

    Parameters
    ----------
    X: type `np.ndarray`, shape (N, M)
        Numpy arrays with N rows, M columns containing the attribute values for N instances
    Y: type `np.ndarray`, shape (N, )
        Numpy arrays with N rows containing the true labels for N instances
        
    Returns the majority label
    """    
    # extract the count from all labels
    label_count: Dict[str, str] = dict()
    for label in Y:
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1

    # get the max label 
    max_label, max_count = "", 0
    for label, count in label_count.items():
        if count > max_count:
            max_label = label
            max_count = count

    return str(max_label)

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
        attribute_names: type List[str]
            A list of column names for each attribute
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

def split(X: np.ndarray, Y: np.ndarray, split_attribute: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """This function takes a dataset and splits it into two sub-datasets according 
    to the values of the split attribute. The left and right values of the split 
    attribute should be in alphabetical order. The left dataset should correspond 
    to the left attribute value, and similarly for the right dataset. 
    
    Parameters
    ----------
    X: type `np.ndarray`, shape (N, M)
        Numpy arrays with N rows, M columns containing the attribute values for N instances
    Y: type `np.ndarray`, shape (N, )
        Numpy arrays with N rows containing the true labels for N instances
    split_attribute: type `int`
        The index of the attribute to split the dataset on

    Returns the tuple of two sub-datasets, i.e. (X_left, Y_left, X_right, Y_right)
    """
    # extract split attribute values and assign left/right according to alphabetical order (i.e. left first)
    values = sorted(list(set([val for val in X[:, split_attribute]])))
    assert len(values) == 2, "Uh oh! You passed an X column that cannot be split. It only has %s unique values." % len(values)
    left_val, right_val = values[0], values[1]

    # split the dataset according to left/right values
    X_left, Y_left, X_right, Y_right = [], [], [], []
    for X_instance, Y_instance in zip(X, Y):
        if X_instance[split_attribute] == left_val:
            X_left.append(X_instance)
            Y_left.append(Y_instance)
        else:
            X_right.append(X_instance)
            Y_right.append(Y_instance)

    return np.array(X_left), np.array(Y_left), np.array(X_right), np.array(Y_right)

def error_rate(Y: np.ndarray, Y_pred: np.ndarray) -> float:
    """This function computes the error rate (i.e. number of incorrectly predicted
    instances divided by total number of instances)

    Parameters
    ----------
    Y: type `np.ndarray`, shape (N, )
        Numpy arrays with N rows containing the true labels for N instances
    Y_pred: type `np.ndarray`, shape (N, )
        Numpy arrays with N rows containing the predicted labels for N instances

    Returns the error rate, which is a float value between 0 and 1
    """    
    incorrect: int = 0
    total: int = Y.shape[0]
    for y, y_pred in zip(Y, Y_pred): 
        if y != y_pred: incorrect += 1
    
    return incorrect / total


class Node: 
    """Data object that helps you build your tree

    Parameters
    ----------
    X: type `np.ndarray`
        Full dataset of all attributes, same as X array returned from load_data
    Y: type `np.ndarray`
        Full dataset of all labels, same as Y array returned from load_data
    attribute_names: type `List`
        List of attribute names, same as attribute_names returned from load_data
    label_names: type `List`
        List of label names, contains the string names of the two unique classes of labels  
    """ 
    def __init__(self, X: np.ndarray, Y: np.ndarray, attribute_names: List[str], label_names: List[str]):
        self.X: np.ndarray = X 
        self.Y: np.ndarray = Y 
        self.attribute_names: List[str] = attribute_names 
        self.label_names: List[str] = label_names 
        
        # additional attributes to be populated when building the decision tree
        self.left: Node = None         # left child of the current node
        self.right: Node = None        # right child of the current node
        self.split_index: int = None   # index of the attribute on which the data is split
        self.predicted_label: str = "" # the label that is assigned to an observation that falls to a leaf node

