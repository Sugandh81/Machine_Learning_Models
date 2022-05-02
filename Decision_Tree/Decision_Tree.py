#!/usr/bin/env python
# coding: utf-8

# 

# 

# In[159]:


from typing import List, Set, Dict, Tuple, Any

import numpy as np
import pandas as pd
import pickle
import sys

TRAIN_FILE = './data/train.csv'   # filepath of train dataset
TEST_FILE = './data/test.csv'     # filepath of test dataset


# The implementation of following function is included in utils files.
# * load_data
# * split
# * error_rate
# 
# For more information please refer "Decision_Tree_Single_Node" folder.

# In[160]:



# importing above functions 
from utils import *


# ### Introduction to Node Object
# 
# This is a node object that is pre-defined  to use to build decision tree. 

# In[161]:


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
        self.predicted_label: str = "" # the label that is assigned to an observation that f


# In[162]:


# Run this cell to access the tree objects for printing
myFirstTree = pickle.load(open('./data/myTree.p', 'rb'))
mySecondTree = pickle.load(open('./data/mySecondTree.p', 'rb'))
myThirdTree = pickle.load(open('./data/myThirdTree.p', 'rb'))


# ###  Printing a Decision Tree 
# 
# 
# Print the decision tree object `myFirstTree`. We will use vertical bars `|` to designate the depth of the node (note that the depth of the root node = 0). The printed tree should look like the following:
# 
#     [15/democrat 13/republican]
#     |  Anti_satellite_test_ban = n: [2 democrat/12 republican]
#     |  |  Export_south_africa = n: [0 democrat/5 republican]
#     |  |  Export_south_africa = y: [2 democrat/7 republican]
#     |  Anti_satellite_test_ban = y: [13 democrat/1 republican]
#     |  |  Export_south_africa = n: [0 democrat/1 republican]
#     |  |  Export_south_africa = y: [13 democrat/0 republican]
# 
# Note: start always with the left child, which will have the split value that comes first alphabetically or numerically. For example, in the tree above, the first node printed is the left node, with data split on the attribute "Anti_satellite_test_ban", and all data for this attribute has the value "n". Then the right node, splitting on the same attribute, contains data where the value is "y"
# 
# 

# In[177]:


def tree_print(current_node: Node, current_depth: int = 0) -> None:

    if (current_node == None):
        return None
    else:
        feat_names, feat_counts = np.unique(current_node.X, return_counts = True)
        label_0_count = sum([i==current_node.label_names[0] for i in current_node.Y])
        label_1_count = sum([i==current_node.label_names[1] for i in current_node.Y])
        
        if current_depth == 0: #if at the root node, data has not been split
            print('[' + str(label_0_count) + '/' + str(current_node.label_names[0]) + 
                  ' ' + str(label_1_count) + '/' + str(current_node.label_names[1]) + ']')
      
            current_depth = current_depth + 1

            tree_print(current_node.left, current_depth)
            tree_print(current_node.right, current_depth)
        else:
            split_val = current_node.X[0,current_node.split_index] #Find the string of the attribute value
            split_name = current_node.attribute_names[current_node.split_index] #Find the string name of the split attribute

            print("|  "*current_depth, end = "")
            print(str(split_name) + ' = ' + str(split_val) + ': ' +
                '[' + str(label_0_count) + ' ' + str(current_node.label_names[0]) + 
                  '/' + str(label_1_count) + ' ' + str(current_node.label_names[1]) + ']')
            current_depth = current_depth + 1
            tree_print(current_node.left, current_depth)
            tree_print(current_node.right, current_depth)
            
myFirstTree = pickle.load(open('./data/myTree.p', 'rb'))


print('\n' + 'Here is the correctly printed first tree:' + '\n')
tree_print(myFirstTree)


# ### Question 2: Computing Entropy [10 pts]
# 
# We start by implementing the `entropy` function that takes in the label column `Y` and output the entropy $H(Y)$, which is defined as:
# 
# $$H(Y) = - \sum_{y \in \mathcal{Y}} P(Y=y) \log_2 P(Y=y)$$
# 
# Empirically, we can compute the entropy as follows:
# 
# $$H(Y) = - \sum_{y \in \mathcal{Y}} \frac{N_{Y=y}}{N} \log_2 \frac{N_{Y=y}}{N}$$
# 
# where $N$ is the total number of instances in the dataset, and $N_{Y=y}$ is the number of instances with label $y$

# In[164]:


def entropy(Y: np.ndarray) -> float:
    """Compute the entropy H(Y)

    Parameters
    ----------
    Y: type `np.ndarray`, shape (N,)
        1D array of the labels

    Returns the entropy H(Y)
    """
    ### YOUR CODE HERE
    labels,label_count = np.unique(Y,return_counts = True)
  
    
    N = len(Y)
    H = 0
    for Ny in label_count:
        prob = Ny/N
        H += -(prob *np.log2(prob))

    return H #replace with proper return statement


# In[165]:


# Create your own datasets to test/debug your entropy function
Y = np.array(['1', '1', '1', '0']) 
print('Your entropy is %s.' % entropy(Y)) # entropy should equal 0.8


# ### Question 3: Computing Conditional Entropy [10 pts]
# Implement the function `conditional_entropy` that computes the conditional entropy $H(Y \mid X_m)$, given the label column `Y` and attribute column `X_m` of attribute $m$:
# 
# $$H(Y \mid X_m) = \sum_{x \in \mathcal{X_m}} P(X_m=x) H(Y \mid X_m=x)$$
# 
# where $H(Y \mid X_m=x)$ is the entropy of all the $Y$ with attribute $m$ equals value $x$: 
# 
# $$H(Y \mid X_m=x) = - \sum_{y \in \mathcal{Y}} P(Y=y | X=x) \log_2 P(Y=y | X=x)$$
# 
# Empirically, since $X_m$ takes in binary values `'0'` and `'1'` in this assignment, we then have:
# 
# $$H(Y \mid X_m) = \frac{N_{X_m=0}}{N} H(Y \mid X_m=0) + \frac{N_{X_m=1}}{N} H(Y \mid X_m=1)$$
# 

# In[166]:


def conditional_entropy(X_m: np.ndarray, Y: np.ndarray) -> float:
    """Compute the conditional entropy H(Y|X_m)

    Parameters
    ----------
    X_m: type `np.ndarray`, shape (N,)
        1D array of attribute m
    Y: type `np.ndarray`, shape (N,)
        1D array of the labels

    Returns the conditional entropy H(Y|X_m)
    """
    ### YOUR CODE HERE
    Xm_val,Xm_count = np.unique(X_m, return_counts = True)
    
    N = X_m.shape[0]
    H = 0
    for i, x in enumerate(Xm_val):
        Y_Xmx = [Y[i] for i in range(N) if X_m[i] == x]
          
                
        H_Y_given_x = entropy(Y_Xmx)
        H += Xm_count[i]/N * H_Y_given_x

    return H 


# In[167]:


#  Create your own datasets to test/debug your conditiona_entropy function
X = np.array(['0', '0', '1', '1'])
Y = np.array(['1', '1', '1', '0'])
print('Your conditional entropy is %s.' % conditional_entropy(X, Y)) # expected answer is 0.5


# ### Computing Mutual Information [10 pts]
# 
# Using `entropy` and `conditional_entropy` implemented above,implement the function `mutual_information` that computes the mutual information-- our choosen metric for choosing the best split in our decision tree. Recall that for label column `Y` and attribute column `X_m` of attribute $m$, the mutual information $I(Y;X_m)$ is:
# 
# $$I(X;Y) = H(Y) - H(Y|X)$$

# In[168]:


def mutual_information(X_m: np.ndarray, Y: np.ndarray) -> float:
    """Calculate mutual information from a single feature column X_m and the label data Y

    Parameters
    ----------
    X_m: np.ndarray, shape (N,)
        1D array of an attribute column index m
    Y: np.ndarray, shape (N,)
        1D array of the labels

    Returns mutual information I(Y; X_m)
    """
    HY_entropy = entropy(Y)
    HY_entropy_given_X = conditional_entropy(X_m, Y)
    mutual_info = HY_entropy - HY_entropy_given_X 

    return mutual_info


# In[169]:


# create your own datasets to test/debug your calc_mi function
X = np.array(['0', '0', '1', '1'])
Y = np.array(['1', '1', '1', '0'])
print('Your mutual info is %s.' % mutual_information(X ,Y)) # expected MI is 0.811 - 0.5 = 0.311


# ###  Finding Attribute with Maximum Mutual Information
# 
# 
# Implementing a function `find_best_attribute` that takes in data `X`, `Y`, then returns the column index (an integer) with the highest mutual information. 
# 
# **This function must "break ties" in the situation where two attributes have equal mutual information. While there are many ways to do this, implement your function such that the attribute with the smaller index is chosen (*i.e.* if index 0 and index 1 have the same MI, return 0, if index 1 and index 3 have the same MI, return 1, etc)**
# 
# 

# In[170]:


def find_best_attribute(X: np.ndarray, Y: np.ndarray) -> int:
    """Find attribute with the greatest mutual information of the dataset X, Y

    Parameters
    ----------
    X: np.ndarray, shape (N, M)
        2D data with M columns as attributes, N rows as instances
    Y: np.ndarray, shape (N,)
        1D array of the labels

    Returns the index of the attribute with greatest mutual information
    """
    bestAttr = 0
    best_mi = 0
    for i, xm in enumerate(X.T):
        mi = mutual_information(xm,Y)
        if best_mi < mi:
            best_mi = mi
            bestAttr = i

    return bestAttr #replace with proper return statement


# In[171]:


# create your own datasets to test/debug your find_best_attribute function
X = np.array([['0', '0'], 
              ['0', '0'],
              ['0', '1'],
              ['0', '1']]) # edit as desired! Remember to keep them as arrays
Y = np.array(['1', '1', '0', '0']) # for this example dataset, max MI should be from X[:,1], return 1

print('The attribute that yields the maximum mutual info is %s.' % find_best_attribute(X, Y)) # expected output is 1


# ###  Training a Decision Tree 
# 
# Now that we have build the necessary components, it's time to build a decision tree!
# 
# Implement a function `train` that takes in all the data `X, Y`, attribute names `attribute_names`, and a maximum depth `max_depth` to build a decision tree using the `Node` structure and recursion

# In[172]:


def train(X: np.ndarray, Y: np.ndarray, attribute_names: List[str], max_depth: int) -> Node: 
    """Build a decision tree with training dataset (X, Y)

    Parameters
    ----------
    X: np.ndarray, shape (N, M)
        2D data with M columns as attributes, N rows as instances
    Y: np.ndarray, shape (N,)
        1D array of the labels
        
    Returns the root node of the built decision tree
    """
    N, M = X.shape
    # if max_depth > number of attributes, set the max depth of the tree to the number of features 
    if max_depth > M:
        max_depth = M
    # find label_names, the list of strings that specifies the two unique classes of Y
    label_names = list(np.unique(Y))

    # instantiate a new Node() to be the root of your tree, 
    
    root = Node(X,Y,attribute_names,label_names) 
    
    
    # call recursive function train_tree to build our decision tree
    return train_tree(node=root, depth=0, max_depth=max_depth)

def train_tree(node: Node, depth: int, max_depth: int) -> Node:
    """This function helps build the decision tree recursively
    
    Parameters
    ----------
    node: type Node
        The current node
    depth: type int 
        The current depth of the current node
    max_depth: type int
        The max_depth
    """
  
  
    # base case 1: if depth >= max_depth, get predicted_label from majority vote, and return node
    if depth >= max_depth:
        node.predicted_label = majority_vote(X= node.X ,Y = node.Y )
        return node
    # base case 2: if X, Y is perfectly classfied, get predicted_labeled from Y, and return node
    elif len(np.unique(Y)) == 1:
        node.predicted_label = Y[0]
        return node
    # recursive case: 
    else:
        # find best attribute from the current data
        
        best_attribute = find_best_attribute(X = node.X,Y = node.Y)
        # check if data can be split into non-empty datasets
        # if not, get predicted_label from majority_vote, and return node
        # print(node)
        if len(np.unique(node.X[:,best_attribute])) < 2:
            node.predicted_label = majority_vote(X = node.X, Y = node.Y)
            return node
        # split the dataset 
        (X_left, Y_left,X_right,Y_right) = split(node.X,node.Y,best_attribute)
        # create left node with the left dataset, set split_index of child to the best attribute
        left_node = Node(X_left,
                         Y_left,
                         node.attribute_names,
                         node.label_names)
        left_node.split_index = best_attribute
    
        # create right node with the right dataset, set split_index of child to the best attribute
        right_node = Node(X_right,
                          Y_right,
                          node.attribute_names,
                          node.label_names)
        right_node.split_index = best_attribute
    
        # increment the depth, recurse on each child node
        node.left = train_tree(left_node, depth + 1, max_depth)
        node.right = train_tree(right_node, depth + 1, max_depth)
        # and set node.left and node.right to their respective child nodes
        
        return node 


# In[173]:


# create your own datasets to test/debug your train function
X = np.array([['0', '0'], 
              ['0', '0'],
              ['0', '1'],
              ['0', '1']]) # edit as desired! Remember to keep them as arrays

Y = np.array(['1', '1', '0', '0'])              
attribute_names = ["zero", "one"]
max_depth = 2
myRoot = train(X,Y, attribute_names,max_depth) # call your train function on the dataset
tree_print(myRoot) # use your tree_print function to visualize your tree!

# For the given example dataset, your trained tree should look like this:
# [2/0 2/1]
# |  one = 0: [0 0/2 1]
# |  one = 1: [2 0/0 1]


# ###  Predicting with a Decision Tree 
# 
# Now that we have a trained decision tree, it's time to use to to make predictions! 

# In[174]:


def predict(trained_tree: Node, X_test: np.ndarray, Y_test: np.ndarray) -> np.ndarray:
    """Traverse the trained tree down the correct path for each observation 
    and return a 1D array of all predicted labels

    Parameters
    ----------
    trained_tree: Node
        A trained decision tree
    X: np.ndarray, shape (N, M)
        2D data with M columns as attributes, N rows as instances
    Y: np.ndarray, shape (N,)
        1D array of the labels

    Returns an numpy array of predicted labels
    """
    N = Y_test.shape[0]
    all_labels = []
    
    # for each instance
    for i in range(N):
        predicted_label = traverse_tree(trained_tree,X_test[i])
        all_labels.append(predicted_label)
        
        # get predicted label by calling traverse_tree, and add to all_labels
        
    return np.array(all_labels)


def traverse_tree(node: Node, x: np.ndarray) -> str:
    """Get the predicted label from a single instance x
    
    Parameters
    ----------
    node: Node 
        current node of the trained decision tree
    x: np.ndarray, shape (M,)
        Attribute vector of a single instance
        
    Returns the predicted label, type `str`
    """
    # base case: if leaf node, then return predicted label 
    if node.left == None and node.right == None:
        return node.predicted_label
     
    # recursive case:
    else:
        split_attribute = node.left.split_index
        
        
        # get the split attribute, which is stored at the split_index of the left 
        # (or right child) of current node 
        
        # get the value of x at the split attribute, call this x_m
        x_m = x[split_attribute]
        # look up which value of X is in the left node, so that we know which branch to traverse 
        
        # call this x_m_left
        x_m_left = node.left.X[0,split_attribute]

        # if x_m is equal to x_m_left, then traverse left and return the label from the left branch
        if x_m == x_m_left:
            return traverse_tree(node = node.left, x = x)
        # else traverse right and return the label from the right branch
        else:
            return traverse_tree(node = node.right, x = x)
            


# In[175]:



def train_and_test(train_filename: str, test_filename: str, max_depth: int) -> Dict[str, Any]: 
    """train_and_test function takes in a train and test dataset, trains a new decision tree,
     performs predictions on train and test, and returns the appropriate outputs

    Parameters
    ----------
    train_filename: type `str`
        The filepath to the training file
    test_filename: type `str`
        The filepath to the testing file
    max_depth: type 'int'
        Desired maximum depth of decision tree

    Returns a dictionary containing the learned decision tree, 
    the train error rate, and the test error rate
    """
    ### YOUR CODE HERE
    X_train, Y_train,attribute_names = load_data(train_filename)
    X_test, Y_test, attribute_names = load_data(test_filename)
    trained_tree = train(X_train, Y_train,attribute_names, max_depth)
    y_pred_train = predict(trained_tree, X_train,Y_train)
    y_pred_test = predict(trained_tree, X_test,Y_test)
    
    train_error = error_rate(y_pred_train,Y_train)
    test_error = error_rate(y_pred_test, Y_test)



    return {
        'tree'       : trained_tree, 
        'train_error': train_error, 
        'test_error' : test_error  
    }


# In[176]:




# full dataset
full_output = train_and_test(TRAIN_FILE, TEST_FILE, max_depth=2)
print('Your train error on the full dataset is: %s' % full_output['train_error']) 
print('Your test error on the full dataset is: %s' % full_output['test_error']) 
print('Your tree trained on full dataset:')
#tree_print(full_output) # uncomment this line to print the learned tree


# In[ ]:





# In[ ]:





# In[ ]:




