#!/usr/bin/env python
# coding: utf-8

# In[19]:


from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import csv


TRAIN_FILE = './data/train.csv'   # filepath of train dataset
TEST_FILE = './data/test.csv'     # filepath of dataset


# In[20]:


def load_data(filename: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """This function takes in the filepath of the data and outputs the tuple of 
    (X, Y, attribute_names). This reader assumes the label Y is always positioned 
    at the last column

    Parameters
    ----------
    filename: type `str`
        The filepath to the dataset

    Returns
    -------
    A tuple (X, Y, attributes_name) where
        X: type `np.ndarray`, shape (N, M)
            Numpy arrays with N rows, M columns containing the attribute values for N training instances
        Y: type `np.ndarray`, shape (N, )
            Numpy arrays with N rows containing the true labels for N training instances
        attribute_names: type `List[str]`
            Names of the attributes
    """
    X: List[str] = []
    Y: List[str] = []
    attribute_names: List[str] = []
    with open(filename, 'r') as f: 
        reader = csv.reader(f)

        # get attribute list, get rid of label header
        attribute_names = next(reader)[:-1] 

        # get X, Y values and convert them to numpy arrays
        for row in reader: 
            X.append(row[:-1])
            Y.append(row[-1])
        X = np.array(X)
        Y = np.array(Y)

    return (X, Y, attribute_names)


# ### Majority Vote Algorithm
# 'Majority Vote' Algorithm takes in X,Y Numpy arrays and outputs desired label. If there are equal number of labels. the tie-breaker goes to the label value that appers first in the dataset.(i,e if there is an equal number of '0' and '1' appears first, then 'majority_vote' outputs to '1'.
# 

# In[21]:



def majority_vote(X: np.ndarray, Y: np.ndarray) -> str:
   """
   Parameters
   ----------
   X: type `np.ndarray`, shape (N, M)
       Numpy arrays with N rows, M columns containing the attribute values for N instances
       
   Y: type `np.ndarray`, shape (N, )
       Numpy arrays with N rows containing the true labels for N instances

   Returns the majority label
   """
   label_count =dict()
   for label in Y:
       if label not in label_count:
           label_count[label] = 0
       label_count[label] += 1
   maxLabel = ''
   maxCount = 0
   for label,count in label_count.items():
       if count > maxCount:
           maxLabel = label
           maxCount = count
   
   return str(maxLabel)
   
   


# ### Dataset Split
# The split function takes in dataset(X,Y)whic is respresented as array, the attribute(respresnted as column index) to split on and returns tuple of split datasets.

# In[22]:


def split(X: np.ndarray, 
          Y: np.ndarray, 
          split_attribute: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    """
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
    
    values = sorted(list(set([val for val in X[:, split_attribute]])))
    left_val, right_val = values[0], values[1]

    X_left = []
    Y_left = []
    X_right = []
    Y_right = []
    
    for X_instance, Y_instance in zip(X, Y):
        if X_instance[split_attribute] == left_val:
            X_left.append(X_instance)
            Y_left.append(Y_instance)
        else:
            X_right.append(X_instance)
            Y_right.append(Y_instance)


    X_left = np.array(X_left)
    Y_left = np.array(Y_left)
    X_right = np.array(X_right)
    Y_right = np.array(Y_right)
    
    return (X_left, Y_left, X_right, Y_right)


# ### Train
# This  function takes in train dataset X_train, Y_train, and the index of the split attribute 'attribute_index'. The output of this function is a tuple of two strings, where the element of the tuple is the label of the left child node, and the second element is the label of the right child node.

# In[23]:


def train(X_train: np.ndarray, Y_train: np.ndarray, attribute_index: int) -> Tuple[str, str]:
    """
    Parameters
    ----------
    X_train: type `np.ndarray`, shape (N, M)
        Numpy arrays with N rows, M columns containing the attribute values for N training instances
    Y_train: type `np.ndarray`, shape (N, )
        Numpy arrays with N rows containing the true labels for N training instances
    attribute_index: type `int`
        The index of the attribute to split the dataset on

    Returns the tuple of labels, i.e. (left_label, right_label)
    """
    
    X_left, Y_left, X_right, Y_right = split(X_train, Y_train, attribute_index)
    left_label = majority_vote(X_left, Y_left)
    right_label = majority_vote(X_right, Y_right)
    
    return (left_label, right_label)


# ### Predict 
# 
# Implement the `predict` function that takes in your trained stump (output of the `train` function), an `X` array, the split `attribute_index` and predicts the labels of `X`. The output should be a 1-D NumPy array with the same number of elements as instances in `X`.
# 

# In[24]:


def predict(left_label: str, right_label: str, X: np.ndarray, attribute_index: int) -> np.ndarray:
    """
    Parameters
    ----------
    left_label: type `str`
        The label corresponds to the left leaf node of the decision stump
    right_label: type `str`
        The label corresponds to the right leaf node of the decision stump
    X: type `np.ndarray`, shape (N, M)
        Numpy arrays with N rows, M columns containing the attribute values for N instances
    attribute_index: type `int`
        The index of the attribute to split the dataset on

    Returns the numpy arrays with shape (N,) containing the label predictions for X
    """
    Y_instance = []
    for X_instance in X:
        if X_instance[0] == '0':
            Y_instance.append(left_label)
        else :
            Y_instance.append(right_label)
    
    Y_instance = np.array(Y_instance)
    return  Y_instance 


# ### Question 8: Error Rate [10 pts]
# 
# Implement the `error_rate` function that takes in the true `Y` values and the `Y_pred` predictions (output of `predict` function) and computes the error rate, which is the number of incorrect instances divided by the number of total instances.

# In[25]:


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
    ### YOUR CODE HERE 
    error_count = 0
    total_count = 0
    for y,y_pred in zip(Y,Y_pred):
        if y != y_pred:
            error_count = error_count + 1
        total_count = total_count + 1
    error_rate = float(error_count/total_count)
    return error_rate # replace this line with your return statement  


# ### Train and Test
# * Use the function `load_data` to load in the training and testing datasets 
# * Use the function `train` to train your decision stump on the training data
# * Use the function `predict` to get the training and testing predictions, with your trained decision stump
# * Use the function `error_rate` to get the train and test error rates

# In[26]:


def train_and_test(train_filename: str, test_filename: str, attribute_index: int) -> Dict[str, Any]:
    """
    Parameters
    ----------
    train_filename: type `str`
        The filepath to the training file
    test_filename: type `str`
        The filepath to the testing file
    attribute_index: type `int`
        The index of the attribute to split the dataset on

    Returns an output dictionary
    """
    split_attribute = 0 
    
    X_train, Y_train, attribute_names = load_data(train_filename)# please complete
    X_test, Y_test, _ = load_data(test_filename)# please complete

    left_label, right_label = train(X_train, Y_train, attribute_index=split_attribute) # please complete
    
    Y_pred_train = predict(left_label, right_label, X_train, attribute_index=split_attribute)
    Y_pred_test = predict(left_label, right_label, X_test, attribute_index=split_attribute)


    train_error_rate = error_rate(Y_train, Y_pred_train)
    test_error_rate = error_rate(Y_test, Y_pred_test) # please complete

    return {
        'attribute_names' : attribute_names,
        'stump'           : (left_label, right_label),
        'train_error_rate': train_error_rate,
        'test_error_rate' : test_error_rate
    }


# In[27]:


train_file = TRAIN_FILE 
test_file = TEST_FILE
attribute_index = 0

# call your function
output_dict = train_and_test(train_file, test_file, attribute_index=attribute_index)

# print the result
print('attributes: ', output_dict['attribute_names'])
print('stump: ', output_dict['stump'])  
print('train error: ', output_dict['train_error_rate']) 
print('test error: ', output_dict['test_error_rate']) 


# ### Summary
# Below we show you the visualization of the decision stump from the previous code block, using the default value (full dataset, `attribute_index=0`).
# <img src="./img/stump_conclusion.png" width="400" height="400">
# 
# We also show you the result table of accuracy, which is $1-\text{error_rate}$:
# 
# | Model | Split attribute | Train accuracy (%) | Test accuracy (%) |
# | :--- | :--- | :--- | :--- |
# | Decision stump | 0 | 70% | 65.8%  |

# ### Key Take away
# Coonstruct a decision stump with a split on one node and use majority vote algorithm at leaf node.

# In[ ]:




