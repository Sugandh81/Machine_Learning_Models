#!/usr/bin/env python
# coding: utf-8

# In[58]:


from typing import List, Tuple, Any, Dict

import math
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


from utils import *


# In[59]:


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# we load the required data beforehand
# loading the data containing only binary attributes (used in M1 and M2)
BINARY_HEART_TRAIN_FILEPATH = './data/binary_heart_train.csv'
BINARY_HEART_TEST_FILEPATH = './data/binary_heart_test.csv'
binary_X_train, binary_Y_train, binary_attributes = load_data(BINARY_HEART_TRAIN_FILEPATH)
binary_X_test, binary_Y_test, _ = load_data(BINARY_HEART_TEST_FILEPATH)

# loading the full dataset with continuous features
HEART_TRAIN_FILEPATH = './data/heart_train.csv'
HEART_TEST_FILEPATH = './data/heart_test.csv'
X_train, Y_train, attributes = load_data(HEART_TRAIN_FILEPATH)
X_test, Y_test, _ = load_data(HEART_TEST_FILEPATH)


# ### Using scikit-learn DecisionTreeClassifier
# 
# For this implementation, we will use the scikit-learn library's robust implementation of [Decision Tree](#https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#). A few important notes:
# - To define a decision tree, call the object `DecisionTreeClassifier(...)` 
#     - *e.g.* `dtree = DecisionTreeClassifier(...)`. 
# - Some parameters of `DecisionTreeClassifier()` that you should take note of:
#     - `criterion`: this is the splitting criterion at each node. Popular choices include 'entropy' and 'gini'; so far we have been using 'entropy' (used to compute mutual information). 
#     - `random_state`: this parameter controls the randomness in the splitting process.  ** set `random_state=RANDOM_SEED` when you create DecisionTreeClassifier**. Not doing this can be a source of unwanted errors.
#     - Other relevant parameters such as `max_depth`, `min_samples_leaf` `min_impurity_decrease` will be used.
# - To train your decision tree, call the `fit()` function with the training data     
#     - *e.g.* `dtree = dtree.fit(X_train, Y_train)`.
# - To get predictions of a dataset, call the `predict()` function 
#     - e.g. `Y_test_pred = dtree.predict(X_test)`.
# - To compute error rate, use `error_rate` implemented 
# 

# In[60]:


def train_and_test(train_filepath: str, test_filepath: str) -> Dict[str, Any]:
    """Similar function interface with the last question in M1 and M2. The main difference 
    is we used scikit-learn DecisionTreeClassifier implementation, instead of your 
    custom-built stump/tree.
    
    Parameters
    ----------
    train_filepath: type `str`
        The filepath to the training file
    test_filepath: type `str`
        The filepath to the testing file
        
    Returns an output dictionary
    """

    # load data
    X_train, Y_train, _ = load_data(train_filepath) 
    X_test, Y_test, _ = load_data(test_filepath)

    # create tree
    dtree = DecisionTreeClassifier(criterion='entropy', random_state=RANDOM_SEED)
    
    # training phase
    dtree = dtree.fit(X_train, Y_train)
    
    # get train and test predictions from trained tree
    Y_train_pred = dtree.predict(X_train)
    Y_test_pred = dtree.predict(X_test)

    # compute train, test errors
    train_error = error_rate(Y_train, Y_train_pred)
    test_error = error_rate(Y_test, Y_test_pred)

    return {
        'tree'       : dtree,
        'train_error': train_error, 
        'test_error' : test_error
    }


# In[61]:


binary_output = train_and_test(BINARY_HEART_TRAIN_FILEPATH, BINARY_HEART_TEST_FILEPATH)
print('train error = %0.2f' %  binary_output['train_error'])
print('test error = %0.2f' %  binary_output['test_error'])


# ### Visualizing your tree
# To help  with debugging and analysis, we implemented a `visualize_tree` function that helps you visualize your output tree. `visualize_tree` is essentially a wrapper around scikit-learn tree visualizations, of which there are two main types:
# - [Text representation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_text.html#sklearn.tree.export_text):  Scikit-learn uses inequalities to denote the split (*e.g.* `smoking = 0` and `smoking = 1` are denoted as `smoking <= 0.5` and `smoking > 0.5`, respectively).
# - [Image visualization](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html): we output an image of our decision tree. Each node in this image is annotated with the split attribute, split threshold, and number of instances at the current node. We also use color to denote the majority labels at each node: orange is 0, and 1 is blue. The darker the color, the more instances that belong to that class are in the node (*e.g.* nodes with dark blue color mean most instances there are labeled 1).
# 
# The `visualize_tree` function is demonstrated below.

# In[62]:


def visualize_tree(dtree: DecisionTreeClassifier, 
                   attributes: List[str], 
                   text: bool = True, 
                   image: bool = True, 
                   fig_size: Tuple[int, int] = (16,10)
                   ) -> None: 
    """Visualizing a trained decision tree of type DecisionTreeClassifier
    
    Parameters
    ----------
    dtree: type `DecisionTreeClassifier`
        A scikit-learn decision tree object
    attributes: type `List[str]`
        List of attribute names in the dataset 
    text: type `bool`, optional
        Indicating whether to visualize text representation
    image: type `bool`, optional
        Indicating whether to output an image of the tree
    fig_size: type `Tuple[int, int]`, optional
        Size of the image, represented by a tuple of (width, height)
    """
    # Text representation
    if text: 
        text_representation = tree.export_text(dtree, feature_names=attributes)
        print(text_representation)
    
    # Image visualization
    if image:          
        fig = plt.figure(figsize=fig_size)
        _ = tree.plot_tree(dtree, 
                           feature_names=attributes,
                           fontsize=10,
                           filled=True)
        plt.show()


# In[63]:


visualize_tree(binary_output['tree'], binary_attributes, text=True, image=True, fig_size=(25, 10))


# ### Decision Tree with Continuous Attributes
# 
# As you can see above, the results from training a decision tree on our binary *heartFailure* dataset didn't yield very good results: the train error is $0.27$ (*i.e.* $73\%$ train accuracy) and the test error is $0.41$ (*i.e.* $59\%$ test accuracy).
# 
# While assuming the dataset to only contain binary attributes is instructive for implementing a decision tree from scratch, most datasets also include attributes with continuous values. Not having this information may contribute to the low performance of our algorithm. As such, let's evaluate our decision tree on the full *heartFailure* dataset.
# 
# First, let's visualize the data. 

# In[64]:


train_data = pd.read_csv(HEART_TRAIN_FILEPATH) # load data into a panda DataFrame, for visualization purpose only
print(train_data.head(10))                     # show top 10 instances


# Now let's see how our decision tree performs on this full dataset.

# In[65]:


full_output = train_and_test(HEART_TRAIN_FILEPATH, HEART_TEST_FILEPATH)
print('train error = %0.2f' %  full_output['train_error'])
print('test error = %0.2f' %  full_output['test_error'])


# Our tree on the full heartFailure dataset performed much better than on binary heartFailure dataset, with 0 training error (i.e.  100%
# 100
# %
#   train accuracy) and  0.31
# 0.31
#  testing error (i.e.  69%
# 69
# %
#   test accuracy).
# Below we visualize the learned tree.

# In[66]:


visualize_tree(full_output['tree'], attributes, text=False, image=True, fig_size=(20, 10))


# One thing to note from the visualization is that, since we are dealing with continuous attributes, we can split on the same attribute again using a different splitting value. For example, following the left branch, we split on `serum_sodium <= 133.5` at depth 2, and `serum_sodium <= 137.5` at depth 4.

# # Part II: Hyperparameter Optimization - Tree Depth
# 
# So far, the decision tree that we implemented was trained on the training dataset and evaluated on test dataset. Hyperparameter optimization is often done experimentally on a "held-out" *validation dataset*, which is a subset of the training dataset. 
# 
# <pre>
# |------------(original) train dataset--------------|
# |-----(new) train dataset + validation dataset-----||-----test dataset (unseen data)---|
# </pre>
# 
# Having a validation dataset allows us to reserve the test dataset for the last step of ML experiments, where we evaluate the model *after* making all the model decisions. For more details, please review lecture videos 6-8, Module 3.
# 
# In this section,we will: 
# - Implement the training and validation pipeline 
# - Analyse train and validation errors vs. tree depth 
# - Visualize the best tree obtained on the validation set, and evaluate it on the test set 
# 
# Run the following cell to split the training data into training and validation sets (please don't mind the function name `train_test_split` -- we simply use this convenient scikit-learn function to split a dataset into two sub-datasets.

# In[67]:


# split train dataset into train and validation set, with validation size being 0.25 of the original train dataset
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, random_state=RANDOM_SEED, test_size=0.25)


# In[68]:


def train_and_val(X_train: np.ndarray, 
                  Y_train: np.ndarray, 
                  X_val: np.ndarray, 
                  Y_val: np.ndarray, 
                  max_depth: int
                  ) -> Dict[str, Any]:
    """Train a decision tree of max_depth on (X_train, Y_train) and evaluate on (X_val, Y_val)
    
    Parameters
    ----------
    X_train: type `np.ndarray`, shape (N_train, M)
        2D NumPy array of train data, with each row represents an instance and each column represents an attribute
    Y_train: type `np.ndarray`, shape (N_train,)
        1D NumPy array of train label
    X_val: type `np.ndarray`, shape (N_val, M)
        2D NumPy array of val data
    Y_val: type `np.ndarray`, shape (N_val,)
        1D NumPy array of val label
    max_depth: type `int`
        The maximum depth of the tree
        
    Returns the output dictionary
    """
    ### YOUR CODE HERE
    dTree = DecisionTreeClassifier(criterion = 'entropy',random_state = RANDOM_SEED,max_depth = max_depth)
    dTree.fit(X_train, Y_train)
    
    trainPred = dTree.predict(X_train)
    testPred = dTree.predict(X_val)
    
    trainError = error_rate(Y_train, trainPred)
    valError = error_rate(Y_val, testPred)
    return {
        'tree'       : dTree, # replace with a trained decision tree
        'train_error': trainError, # replace with the train error
        'val_error'  : valError  # replace with the validation error
    } 


# In[69]:


# dataset for unit testing
X_train_unit = np.array([
    [0,1,0,0],
    [1,0,1,0],
    [1,0,0,1],
    [1,0,0,0]
])
Y_train_unit = np.array([
    1,
    1,
    0,
    0
])
X_val_unit = np.array([
    [0,0,0,0],
    [1,1,0,0]
])
Y_val_unit = np.array([
    1,
    0
])

# run the function and visualize the tree
unit_output = train_and_val(X_train_unit, Y_train_unit, X_val_unit, Y_val_unit,  max_depth=1)
print('train error=', unit_output['train_error']) # should be 0.25
print('val_error=', unit_output['val_error']) # should be 0.0
visualize_tree(unit_output['tree'], attributes, text=False)


# ### Question 2: Grid Search on Tree Depth  [15 pts]
# 
# When doing hyperparameter optimization, we often define a range of hyperparameter values on which to train and evaluate our model. Each value will yield a different tree and we want to return the best tree based on the validation result.
# 
# Implement the function `grid_search_on_depth` that takes in the list of possible tree max depths in addition to the train and validation data. This function returns the best tree with the best max depth and the best max depth value itself. In addition, please return lists of train and validation error rate, where each error rate is obtained at a max depth value-- this will be used for plotting. 
# 
# 

# In[70]:


def grid_search_on_depth(X_train: np.ndarray, 
                         Y_train: np.ndarray, 
                         X_val: np.ndarray, 
                         Y_val: np.ndarray, 
                         max_depths: List[int]
                         ) -> Dict[str, Any]:
    """This function takes in the train data, validation data, and list of 
    possible values for tree max depth. We train a decision tree on each 
    depth value and find the best performing tree on the validation set.
    
    Parameters
    ----------
    X_train: type `np.ndarray`, shape (N_train, M)
        2D NumPy array of train data, with each row represents an instance 
        and each column represents an attribute
    Y_train: type `np.ndarray`, shape (N_train,)
        1D NumPy array of train label
    X_val: type `np.ndarray`, shape (N_val, M)
        2D NumPy array of val data
    Y_val: type `np.ndarray`, shape (N_val,)
        1D NumPy array of val label
    max_depths: type `List[int]`
        List of the maximum depths of the tree
        
    Returns a dictionary containing the best tree, best depth, 
    list of train error rates, list of val error rates
    """
    
    # create empty lists for train and val errors
    trainError = []
    valError =[]
    

    # create variables for storing the best validation error, best tree, and best depth 
 
    best_val_error = None
    best_tree = None
    best_depth = None
    
    # loop through each depth d in max_depths 
    for d in max_depths:
        
        results = train_and_val(X_train, Y_train, X_val,Y_val,d)
        trainError.append(results['train_error'])
        valError.append(results['val_error'])
        
        if best_val_error == None:
            best_val_error = valError[-1]
        
        elif valError[-1] < best_val_error:
            best_val_error = valError[-1]
            best_tree = results['tree']
            best_depth = d
    
    return {
        'best_tree'   : best_tree, # replace with the appropriate value
        'best_depth'  : best_depth, # replace with the appropriate value
        'train_errors': trainError, # replace with the appropriate value
        'val_errors'  : valError  # replace with the appropriate value
    }
    


# ###  Plotting and Analysis 
# 
# A useful way to monitor the hyperparameter optimization process is to visualize the train and error rates as a function of the hyperparameter values. As such, implement the `plot` function below that takes in the depth values, best depth, train and validation error rates, and creates the desired plot. 
# 
# 
# - `x`: The vertical coordinates of the data points
# - `y`: The horizontal coordinates of the data points
# - `label`: The name of the curve, to help with legend. This should be either 'training' or 'validation'.
# 
# Your output image file will appear as `error_rates.png` for inspection. 

# In[71]:


def plot(depths: List[int], best_depth: int, train_errors: List[float], val_errors: List[float]) -> None:
    """Plot the train and validation error rates vs. tree max depths
    
    Parameters
    ----------
    depths: type `List[int]`
        List of depth values
    best_depth: type `int`
        Tree depth which gives the lowest validation error rate
    train_errors: type `List[float]`
        List of train error rates, corresponding to each value in depth
    val_errors: type `List[float]`
        List of val error rates, corresponding to each value in depth
    """    
    # create a new figure, to avoid duplicating with figure in plot
    plt.figure()
    
    # add title and labels and title
    plt.title('Error Rates vs. Tree Max Depths')
    plt.xlabel('Tree Max Depths')
    plt.xticks(depths)
    plt.ylabel('Error Rate')
    
    # TODO: add train and val error plots here
    plt.plot(depths,train_errors,label = 'training')
    plt.plot(depths,val_errors,label = 'validation')
    
    # add highlilght best depth on validation data
    depth_idx = depths.index(best_depth)
    plt.plot([best_depth], [val_errors[depth_idx]], 'ro')
    
    # show legends and save figure
    plt.legend() # show legend
    plt.savefig('results/error_rates.png') # save figure for comparison


# Now that the required functions have been implemented, it's now time put them together to analyze the hyperparameter optimization process. 
# 
# Here is a rough outline of the process:
# - Define the hyperparameters and their range of values (tree depth in this case)
# - Train and evaluate the decision tree with each hyperparameter value
# - Monitor with visualizations of plots, best tree, and best validation error rate
# - Select the best model
# - Finally evaluate on test set with the best model
# 
# Run the following cell:

# In[72]:


get_ipython().system('mkdir results')


# In[73]:


# define the max depth and range of values (1 to 10)
max_depth = 10
depths = list(range(1, max_depth))

# train, evaluate, and get the best decision tree 
output = grid_search_on_depth(X_train, Y_train, X_val, Y_val, depths)

# ouput the best validation error, plot visualization, and the corresponding best tree
print('best val error = %0.4f' % min(output['val_errors']))
plot(depths, output['best_depth'], output['train_errors'], output['val_errors'])
visualize_tree(output['best_tree'], attributes, text=False)


# Some initial observations:
# - The validation error rate is $8.9\%$, which means our validation accuracy is around $91.1\%$
# - The error rates vs. depth plots show that the best tree is not necessary the deepest: the best depth is 4
# - The tree visualization shows a more managable tree with depth 4 than the previous [untuned tree](#visualizing)
# 
# Finally, we evaluate the best model on the test data.

# In[74]:


Y_test_pred = output['best_tree'].predict(X_test)
test_error = error_rate(Y_test, Y_test_pred)
print('test error=%0.4f' % test_error) # should be 0.2667


# # Part III: Hyperparameter Optimization - Grid Search
# 
# You have seen the effect of tree depth on error rate. A model often has more than one hyperparameter to optimize. A basic technique to tune these hyperparameters is *grid search*, where models are evaluated on all combinations of hyperparameters and their values based on a pre-defined range for each hyperparameter. 
# 
# In this section, you will implement grid search over the following hyperparameters:  
# - `criterion`: the criterion with which to measure the splitting quality. Two popular options are [Gini impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity) ('gini') and Mutual Information ('entropy')
# - `max_depth`: the maximum depth of the tree
# - `min_samples_leaf`: the minimum number of instances required at each leaf node. For example, if `min_sample_leaf=5`, then at least 5 instances are required at each leaf node.  
# - `min_impurity_decrease`: a node will only be split if the decrease in purity (*e.g.* Gini impurity or Mutual Information) is greater than this 'threshold' value.
# 
# You can set the value of these hyperparameters when creating [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier). The ranges of the hyperparameter depend on the test cases. For the full *heartFailure* dataset, we will be using the following ranges:
# - `criterion`: `gini` or `entropy`
# - `max_depth`: 1-5 
# - `min_samples_leaf`: 1-5
# - `min_impurity_decrease`: 0.1, 0.2, 0.3, 0.4, 0.5

# ### Question 4: Grid search [25 pts]
# 
# Implement `grid_search` that takes in the train dataset, validation dataset, hyperparameters, and returns the best decision tree, best hyperparameter values, and best validation error. The input hyperparameters take the form of a dictionary mapping the hyperparameter names to their value ranges. An example of this dictionary is shown below:
# 
#         {
#             'criteria':  ['gini', 'entropy'],
#             'max_depths': [1, 2, 3],
#             'min_samples_leaves': [1, 2, 3],
#             'min_impurity_decreases': [0.1, 0.2]
#         }
# 
# 
# 
# - To get the values of the hyperparameter, index the dictionary with the (plural) hyperparameters name. For example, `hyperparameters['criteria']` gives you the list `['gini', 'entropy']`
# - In your implementation, you should have a few *for* loops, corresponding to the hyperparameters
# - The [DecisionTreeClassifier object](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier) contains parameters that correspond to the hyperparameters defined above. For example, to create a decision tree using Gini impurity with max depth 2, you can do `dtree = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=RANDOM_SEED)`. The parameters that you need to use for this assignment are `criterion, max_depth, min_samples_leaf, min_impurity_decrease`
# - As a reminder, when creating your DecisionTreeClassifier object, **you must set `random_state` to `RANDOM_SEED`**
# - Please return the best hyperparameters as a tuple of (`best criterion, best max_depth, best min_sample_leaf, best min_impurity_decrease`), such as `('gini', 5, 3, 0.5)`

# In[75]:


def grid_search(X_train: np.ndarray, 
                Y_train: np.ndarray, 
                X_val: np.ndarray, 
                Y_val: np.ndarray, 
                hyperparameters: Dict[str, Any]
                ) -> Dict[str, Any]:
    """Perform grid search on decision tree hyperparameters
    
    Parameters
    ----------
    X_train: type `np.ndarray`
        2D NumPy array of train data, with each row represents an instance and 
        each column represents an attribute
    Y_train: type `np.ndarray`
        1D NumPy array of train label
    X_val: type `np.ndarray`
        2D NumPy array of val data
    Y_val: type `np.ndarray`
        1D NumPy array of val label
    hyperparameters: type `Dict[str, Any]`
        A dictionary mapping the hyperparameter names to their value ranges. 
        An example of this dictionary is shown below
        {
            'criteria':  ['gini', 'entropy'],
            'max_depths': [1, 2, 3]
            'min_samples_leaves': [1, 2, 3],
            'min_impurity_decreases': [0.1, 0.2]
        }
        To get the values of the hyperparameter, you index the dictionary with the hyperparameters name. 
        For example, hyperparameters['criteria'] gives you the list ['gini', 'entropy']
        
    Returns the output dictionary containing best tree, best hyperparameters, and best val error
    """    
   
    
    # create variables for storing the best tree, best val error, and best set of hyperparameters
    best_tree = None
    best_val_error = None
    best_hyperParam = None
    
    
    # for each set of hyperparameters (4 nested for-loops, one for each hyperparameter)
    for c in hyperparameters['criteria']:
        for m in hyperparameters['max_depths']:
            for s in hyperparameters['min_samples_leaves']:
                for i in hyperparameters['min_impurity_decreases']:
                    dTree = DecisionTreeClassifier(criterion = c,
                                                  max_depth = m,
                                                  min_samples_leaf = s,
                                                  min_impurity_decrease = i,
                                                  random_state = RANDOM_SEED)
                    dTree.fit(X_train,Y_train)
                    preds = dTree.predict(X_val)
                    score = error_rate(Y_val,preds)
                    if best_val_error == None:
                        best_val_error = score
                    if score < best_val_error :
                        best_val_error = score
                        best_tree = dTree
                        best_hyperParam = (c, m, s, i)
    
    
    return {
        'best_tree'           : best_tree, # replace with the appropriate value
        'best_hyperparameters': best_hyperParam, # replace with the appropriate value
        'best_val_error'      : best_val_error # replace with the appropriate value
    }


# In[76]:


# do grid search on pre-defined hyperaparameter ranges
h = {
    'criteria': ['entropy', 'gini'], 
    'max_depths': list(range(1, 6)), 
    'min_samples_leaves': list(range(1, 6)), 
    'min_impurity_decreases': [i/10 for i in range(6)]
}
output = grid_search(X_train, Y_train, X_val, Y_val, h)

# visualize the results
print('best val error=%0.4f' % output['best_val_error']) # should be 0.0714
print('best criterion, max_depth, min_samples_leaf, min_impurity_decrease=', output['best_hyperparameters']) # should be ('gini', 2, 1, 0.0)
visualize_tree(output['best_tree'], attributes, text=True)


# In[77]:


# Finally evaluate on test set
Y_test_pred = output['best_tree'].predict(X_test)
test_error = error_rate(Y_test, Y_test_pred)
print('test error=%0.4f' % test_error) # should be 0.2800


# #  K-Fold Cross-Validation 
# 
# Taking a subset of the training data for validation helps us preserve the test set for the very last step of evaluation. When we have a lot of training data, this doesn't present a problem. However, when we don't have a lot of data (like in the *heartFailure* dataset), we are taking away data that can be used for training. 
# 
# A solution for this problem is *K-Fold Cross-Validation*, where we partition the data into $k$ *folds* and we train our model $k$ times. For each run, a fold is used as validation data while the rest are used as training data. Thus we can both evaluate on held-out validation set while not losing valuable training data. 
# 
# The high-level steps for K-Fold Cross-Validation are:
# - Separating the training data into folds
# - For each set of hyperaparameters:
#     - For each of the $k$ runs:
#         - Create training and val set 
#         - Train the tree
#         - Evaluate on val        
# - Return the tree with the best average val error
# 
# **Note**: Here we use *average validation error* (over all $k$ runs) as the performance metric for each set of hyperparameters. 

# ### Create Folds 
# 
# Implement the `create_folds` function that, given training data and $k$, separates the training data into $k$ folds. Some important rules for creating folds in this question: 
# - In case the number of instances is not divisible by $k$, we choose to take the ceiling of the quotient (*i.e.* the number of instances per partition). The last partition will be what is left. 
# 
# 
# 
# 

# In[78]:


def create_folds(X: np.ndarray, 
                 Y: np.ndarray, 
                 k: int
                 ) -> List[Tuple[np.ndarray, np.ndarray]]:    
    """This function separates (X,Y) into k partitions in order. In case the number of instances 
    is not divisible by k, we choose to take ceiling of the quotient (i.e. the number of instances per partition)
        
    Parameters
    ----------
    X: type `np.ndarray`
        2D NumPy array of data, with each row represents an instance and each column represents an attribute
    Y: type `np.ndarray`
        1D NumPy array of label
    k: type `int`
        The given number of partition we want to split the dataset on
        
    Returns list of k folds: [(X_0, Y_0), ..., (X_{k-1}, Y_{k-1})]
    """
    # YOUR CODE HERE
    cv_folds = []
    num_obs = X.shape[0]
    num_per_fold = int(np.ceil(num_obs/k))
    for i in range(0, num_obs, num_per_fold):
        start = i
        end = min(i + num_per_fold, num_obs)
        X_i = X[start:end]
        Y_i = Y[start:end]
        cv_folds.append((X_i,Y_i))
    return cv_folds


# In[79]:


# a dataset that you can play with (also dataset for unit testing)
X = np.array([[1],[0],[1],[0],[1],[0],[1],[0],[1],[0]])
Y = np.array([1,0,1,0,1,0,1,0,1,0])
k = 3
folds = create_folds(X, Y, k)
print(folds)


# ###  Create Cross-Validation Data 
# 
# Implement the `create_cv_data` function that takes in the $k$ data folds as well as the index of the validation fold (`val_index`) and outputs the appropriate training and validation data. `val_index` starts from 0 and does not exceed $k-1$. 
# 
# As an example, if we have $folds = [(x_0, y_0),(x_1, y_1),(x_2, y_2),(x_3, y_3)]$ and `val_index=1`, then our train dataset will be $X_{train}=[x_0, x_2, x_3], Y_{train}=[y_0, y_2, y_3]$, and val dataset is $X_{val}=[x_1], Y_{val}=[y_1]$
# 
# 

# In[80]:


def create_cv_data(folds: List[Tuple[np.ndarray, np.ndarray]], 
                   val_index: int
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create the training and validation data for K-fold Cross-Validation
    
    Parameters
    ----------
    folds: type `List[Tuple[np.ndarray, np.ndarray]]`
        The data folds. If we have k folds, this will be [(X_1, Y_1), (X_2, Y_2),..., (X_k, Y_k)]
    val_index: type `int`
        Index of validation set. If we have k folds, then 0 <= val_index <= k-1
    
    Returns (X_train, Y_train, X_val, Y_val)
    """
    # YOUR CODE HERE
    X_train = []
    Y_train = []
    X_val,Y_val = folds[val_index]
    folds_len = len(folds)
    for i in range(folds_len):
        if i != val_index:
            X, Y = folds[i]
            X_train.append(X)
            Y_train.append(Y)
    X_train = np.concatenate(X_train)
    Y_train = np.concatenate(Y_train)
    
    return (X_train, Y_train, X_val, Y_val) 


# In[81]:


# unit test dataset: 3-element fold
folds_unit = [(np.array([[1]]), np.array([1])), (np.array([[0]]), np.array([0])), (np.array([[1]]), np.array([1]))]
val_index = 0
X_train_unit, Y_train_unit, X_val_unit, Y_val_unit = create_cv_data(folds_unit, val_index)
print('X_train=', X_train_unit) 
print('Y_train=', Y_train_unit) 
print('X_val=', X_val_unit)     
print('Y_val=', Y_val_unit)    


# ###  Put things together
# 
# Using `create_folds` and `create_cv_data` functions above, implement `kfold_cv` that takes in the training dataset, $k$ partitions, hyperparameters, and returns the best decision tree, best hyperparameter values, and best validation error. The input hyperparameters take the form of a dictionary mapping the hyperparameter names to their value ranges, and best hyperparameter values are represented as a tuple of (best criterion, best max_depth, best min_sample_leaf, best min_impurity_decrease), in that order.
# 
# **Important**: Used *average validation error* as the performance metric. This means averaging the validation error over $k$ runs for each set of hyperparameters, and use this number to get the best tree/hyperparameters.
# 
# **A note on efficiency**: running K-Fold Cross-Validation can be *significantly slower* than the previous questions, particularly on the full *heartFailure* dataset. However, we still expect your implementation to run in less than 30 seconds. For reference, our non-optimized implementation takes 2-4 seconds on average.

# In[82]:


## GRADED
### YOUR SOLUTION HERE
def kfold_cv(X_train: np.ndarray, 
             Y_train: np.ndarray, 
             k: int, 
             hyperparameters: Dict[str, Any]
             ) -> Dict[str, Any]:
    """Perform K-fold Cross-Validation to find the best decision tree hyperparameters 
    
    Parameters
    ----------
    X_train: type `np.ndarray`
        2D NumPy array of train data, with each row represents an instance and each column represents an attribute
    Y_train: type `np.ndarray`
        1D NumPy array of train label
    k: type `int`
        The given number of partition we want to split the dataset on
    hyperparameters: type `Dict[str, Any]`
        A dictionary mapping the hyperparameter names to their value ranges. 
        An example of this dictionary is shown below
        {
            'criteria':  ['gini', 'entropy'],
            'max_depths': [1, 2, 3]
            'min_samples_leaves': [1, 2, 3],
            'min_impurity_decreases': [0.1, 0.2]
        }
        To get the values of the hyperparameter, you index the dictionary with the hyperparameters name. 
        For example, hyperparameters['criteria'] gives you the list ['gini', 'entropy']
        
    Returns an output dictionary containing best tree, best hyperparameters, best val error
    """ 
    # YOUR CODE HERE
    # Pseudocode hints: 
    
    # create variables for storing the best tree, best val error, and best set of hyperparameters
    best_hyperparameter = None
    best_tree = None
    best_val_error = None
    
    # create the folds (using create_folds)
     
    folds = create_folds(X_train, Y_train,k)
    # for each set of hyperparameters (4 nested for-loops, one for each hyperparameter)
    for c in hyperparameters['criteria']:
        for m in hyperparameters['max_depths']:
            for s in hyperparameters['min_samples_leaves']:
                for i in hyperparameters['min_impurity_decreases']:
                    val_errors = []
                # for each j in k partitions
                    for j in range(k):
                        X_train, Y_train, X_cval,Y_cval =  create_cv_data(folds,j)
                        dTree = DecisionTreeClassifier(max_depth = m,criterion = c,min_samples_leaf = s,
                                                       min_impurity_decrease = i, random_state = RANDOM_SEED)
                        dTree.fit(X_train,Y_train)
                        score = dTree.predict(X_cval)
                        error_rt = error_rate(Y_cval,score)
                        val_errors.append(error_rt)
                    avg_val_error = sum(val_errors)/k
                    
                    if best_val_error == None:
                        best_val_error = avg_val_error
                    if avg_val_error < best_val_error:
                        best_val_error = avg_val_error
                        best_tree = dTree
                        best_hyperparameter = (c,m,s,i)
                                
            
    return {
        'best_tree'           : best_tree, 
        'best_hyperparameters': best_hyperparameter,
        'best_val_error'      : best_val_error  
    }


# In[83]:


# reload data, k, and hyperparameter config h
X_train, Y_train, _ = load_data(HEART_TRAIN_FILEPATH)
k = 3
h = {
   'criteria': ['entropy', 'gini'], 
   'max_depths': list(range(1, 6)), 
   'min_samples_leaves': list(range(1, 6)), 
   'min_impurity_decreases': [i/10 for i in range(6)]
}

# k-fold cross-validation
output = kfold_cv(X_train, Y_train, k, h)

# output results
print('best val error=%0.4f' % output['best_val_error']) # should be 0.1116
print('best criterion, max_depth, min_samples_leaf, min_impurity_decrease=', output['best_hyperparameters'])
visualize_tree(output['best_tree'], attributes, text=False)

# finally evaluate on test set
Y_test_pred = output['best_tree'].predict(X_test)
test_error = error_rate(Y_test, Y_test_pred)
print('test error=%0.4f' % test_error) # should be 0.2800


# # Summary: Comprehensive Tables
# 
# In this assignment, you have learned to:
# - Perform hyperparameter optimization on the tree depth and analyze its effect on the train/validation error rates
# - Implement and perform grid search on several decision tree hyperparameters
# - Implement and perform K-Fold Cross-Validation on several decision tree hyperparameters
# 
# 
# 

# In[ ]:





# In[ ]:




