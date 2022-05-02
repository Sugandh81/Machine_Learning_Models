#!/usr/bin/env python
# coding: utf-8

# In[1]:


from typing import List, Tuple, Any, Dict

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from testing import TestHelper
from utils import * 

# random seed that helps us deterministically autograde your answers
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# In[2]:


# setting up sample data
sample_X_train = np.array([[1.0], [2.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0], [12.0]])
sample_Y_train = np.array([[0], [0], [0], [0],[1],[0],[1],[1],[1],[1]]).reshape((10,))
sample_X_val = np.array([[1.5], [3.0], [6.0]])
sample_Y_val = np.array([[0], [1], [1]]).reshape((3,))


# In[3]:


def train_and_val(X_train: np.ndarray, 
                  Y_train: np.ndarray, 
                  X_val: np.ndarray, 
                  Y_val: np.ndarray
                  ) -> Dict[str, Any]:
    """Similar function interface as implemented in M6. The main difference 
    is we used scikit-learn LogisticRegression implementation, instead of your custom-built SGD implementation.
    We also return the regression object itself.
    
    Parameters
    ----------
    X_train: type `np.ndarray`
        2D NumPy array of train data, with each row represents an instance and each column represents an attribute
    Y_train: type `np.ndarray`
        1D NumPy array of train label
    X_val: type `np.ndarray`
        2D NumPy array of val data
    Y_val: type `np.ndarray`
        1D NumPy array of val label
          
    Returns a dictionary with relevant results
    """
    # create Logistic Regression model
    logreg = LogisticRegression(penalty='none', random_state=RANDOM_SEED, solver='saga', max_iter=10000)
    
    # training phase
    logreg = logreg.fit(X_train, Y_train)
    
    # get train and test predictions from trained Logistic Regression
    Y_train_pred = logreg.predict(X_train)
    Y_val_pred = logreg.predict(X_val)

    # compute train, test errors
    train_score = logreg.score(X_train, Y_train)
    val_score = logreg.score(X_val, Y_val)

    return {
        'best_model'         : logreg, 
        'Y_train_predictions': Y_train_pred, 
        'Y_val_predictions'  : Y_val_pred, 
        'train_score'        : train_score, 
        'val_score'          : val_score
    }


# In[4]:


sample_output = train_and_val(sample_X_train, sample_Y_train, sample_X_val, sample_Y_val)


# In[5]:


sample_output


# In[6]:


def error_rate(Y: np.ndarray, Y_hat: np.ndarray) -> float:
    """Given the true labels Y and predicted label Y_hat, returns the error rate
    
    Parameters
    ----------
    Y: type `np.ndarray`, shape (N, 1)
        2D numpy array of true labels
    Y_hat: type `np.ndarray`, shape (N, 1)
        2D numpy array of predicted labels
    """
    N = Y.shape[0]
    num_incorrect = 0
    for (Y_i, Y_hat_i) in zip(Y, Y_hat):
        if Y_i != Y_hat_i: num_incorrect += 1
    return num_incorrect / N


# ###  Calculating Error Rate 
# 
# 

# In[9]:


val_error_rate = error_rate(sample_Y_val,sample_output['Y_train_predictions'])
val_error_rate


# In[11]:


print('The validation score from the error_rate function is: %s' % val_error_rate) # should b 0.667
print('The validation score from the score() method in the train_and_val function is: %s' % sample_output['val_score']) # should b 0.333


# ###  Examining the score() Method 
# 
# After looking at the comparison in the validation scores above, reading the [documentation on the `scikit-learn` website](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.score), and perhaps modifing the code to look at the training error as well, what error does the `score()` method return for the LogisticRegression function?
# 
# 1. The Logistic Regression `score()` method returns the $R^2$ score of the best fit line.
# 2. The Logistic Regression `score()` method returns the same error as the `error_rate` function, which is the percent error from the true labels $y$ and the predicted labels $\hat{y}$.
# 3. The Logistic Regression `score()` method produces the percent correct from the true labels $y$ and the predicted labels $\hat{y}$ -- (*i.e.* 1-`error_rate` error).
# 
# 

# # Scikit-learn Logistic Regression for an NLP Dataset
# 
# Now that we have implemented binary logistic regression, let's apply it to predict whether a movie review is negative or positive! In this section, we will run the following cells to import the dataset on the Natural Language Processing (NLP) dataset, namely the [Movie Review Polarity dataset](https://www.kaggle.com/nltkdata/movie-review).

# In[13]:


# debug dataset
DEBUG_FILEPATH = './data/debug/reviews.tsv'
DEBUG_DICT_FILEPATH = './data/debug/dict.txt'

# full dataset
TRAIN_FILEPATH = './data/full/train_data.tsv'
VAL_FILEPATH = './data/full/valid_data.tsv'
TEST_FILEPATH = './data/full/test_data.tsv'
FULL_DICT_FILEPATH = './data/full/dict.txt' 


# In[14]:


with open(DEBUG_FILEPATH, 'r') as f:
    for line in f.readlines():
        print(line)


# In[15]:


print('Loading data...', end='')

# loading data
token2index = load_dict(FULL_DICT_FILEPATH)
X_train, Y_train = load_data(TRAIN_FILEPATH, token2index)
X_val, Y_val = load_data(VAL_FILEPATH, token2index)
X_test, Y_test = load_data(TEST_FILEPATH, token2index)

# reshaping Y for appropriate input into scikit-learn Logistic Regression
Y_train = Y_train.reshape((-1,))
Y_val = Y_val.reshape((-1,))
Y_test = Y_test.reshape((-1,))
print('finished!')


# In[16]:


def test_accuracy(trained_model, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """This function calculates the accuracy for the movie test set from trained logistic regression.

    Parameters
    ---------
    trained_model
        This is the trained LogisticRegression object that was stored in your train_and_val function.
    X_test: type 'np.ndarray'
        2D NumPy array of test data
    Y_test: type 'np.ndarray'
        1D NumPy array of test label
    """ 

    return trained_model.score(X_test,y_test) 


# In[17]:


noreg_output = train_and_val(X_train, Y_train, X_val, Y_val)
print('train accuracy: %s' % noreg_output['train_score'])
print('val accuracy: %s' % noreg_output['val_score'])

noreg_test_accuracy = test_accuracy(noreg_output['best_model'], X_test, Y_test)
print('test accuracy: %s' % noreg_test_accuracy) # should be 0.63


# # Training Logistic Regression with Regularization

# In[19]:


def train_and_val_regularized(X_train: np.ndarray, 
                              Y_train: np.ndarray, 
                              X_val: np.ndarray, 
                              Y_val: np.ndarray, 
                              penalty: str, 
                              C: float
                              ) -> Dict[str, Any]:
    """Similar function interface as implemented earlier in this assignment. The main difference 
    is that you now are including a location to change the penalty term and the C parameter. 
    
    Parameters
    ----------
    X_train: type `np.ndarray`
        2D NumPy array of train data, with each row represents an instance and each column represents an attribute
    Y_train: type `np.ndarray`
        1D NumPy array of train label
    X_val: type `np.ndarray`
        2D NumPy array of val data
    Y_val: type `np.ndarray`
        1D NumPy array of val label
    penalty: type 'str'
        String of 'none' for no regularization, 'l1' for L1-norm regularization, and 'l2' for L2-norm regularization
    C: type 'float'
        Float value for the strength of regularization
        
    Returns the dictionary of relevant output
    """
 
    lgr = LogisticRegression(penalty = penalty,solver = 'liblinear', random_state=RANDOM_SEED).fit(X_train,Y_train)
    return {
        'best_model'         : lgr, 
        'Y_train_predictions': lgr.predict(X_train),  
        'Y_val_predictions'  : lgr.predict(X_val),  
        'train_score'        : lgr.score(X_train,Y_train), 
        'val_score'          : lgr.score(X_val,Y_val)   
    }


# ###  K-fold Cross-Validation on Regularization Strength  
# 
# When doing hyperparameter optimization, we often define a range of hyperparameter values on which to train and evaluate our model. Each value will yield a logistic regression model, and we want to return the best logistic regression based on the K-fold validation.
# 
# We are going to implement the function `grid_search_on_C` using a simple scikit-learn function, `GridSearchCV`. This function returns the best trained model, regularization strength $C$ hyperparameter, and best cross-validation score. We strongly recommend you review the documentation of `GridSearchCV` [here.](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
# 
# 

# In[20]:


def grid_search_on_C(X_train: np.ndarray, 
                     Y_train: np.ndarray, 
                     penalty: str,
                     C_param: List[float],
                     cv: int
                     ) -> Dict[str, Any]:
    """This function takes in the train data and list of possible values for C. 
    We train a GridSearchCV on each C and find the best performing LogisticRegression 
    from the k-folds.
    
    Parameters
    ----------
    X_train: type `np.ndarray`
        2D NumPy array of train data, with each row represents an instance and each 
        column represents an attribute
    Y_train: type `np.ndarray`
        1D NumPy array of train label
    penalty: type 'str'
        String of 'none' for no regularization, 
        'l1' for L1-norm regularization, and 
        'l2' for L2-norm regularization
    C_param: type List[float]
        List of float values for the strength of regularization
    cv: type 'int'
        Int type value for the number of k-folds
        
    Returns a dictionary containing the fitted logistic regression model, the best C, 
    and the score corresponding to the best C
    """
    # set up a LogisticRegression estimator (i.e. model)
    lgr = LogisticRegression(penalty = penalty,solver = 'liblinear',random_state = RANDOM_SEED)
    
    # create a dictionary containing hypeparameter C and its values that you will perform grid search over
    params = {'C':C_param}
    # create GridSearchCV object, inputting model, hyperparameter dictionary, andd cv
    grid = GridSearchCV(lgr,param_grid=params)
    # get the best logreg model best_logreg by fitting the GridSearchCV object with the training data
    grid.fit(X_train,Y_train)
    best_logreg = grid.best_estimator_
    # get the best C by using best_logreg.best_params_
    best_C = grid.best_params_
    # get the best score by using best_logreg.best_params_
    best_score = grid.best_score_

    return {
        'best_model': best_logreg, 
        'best_C'    : best_C, 
        'best_score': best_score 
    }


# ### Analysis of the Best Models
# 
# We will now use `grid_search_on_C` to find the best $C$ hyperparamaters for both 'L1' and 'L2' regularization. We will first set the `C_params` to be tested as a range of values from 0.05 to 1.

# In[21]:


C = np.logspace(-4,5,num=15,base=np.e)
l1_output = grid_search_on_C(X_train=X_train, Y_train=Y_train, penalty='l1', C_param=C, cv=5)
l2_output = grid_search_on_C(X_train=X_train, Y_train=Y_train, penalty='l2', C_param=C, cv=5)
print('L1 best C: %s' % l1_output['best_C'])
print('L2 best C: %s' % l2_output['best_C'])


# Now you will use the best $C$ found in the cell above andd compare the coefficients corresponding to the best fit parameters for 'L1' and 'L2' regularization. Run the following cell to calculate the train, validation, and test scores for the 'L2' regularization

# In[22]:


l2_model = train_and_val_regularized(X_train, Y_train, X_val, Y_val, 'l2', 0.03) 
l2_test_accuracy = test_accuracy(l2_model['best_model'], X_test, Y_test)
print('train accuracy: %s' % l2_model['train_score'])
print('val accuracy: %s' % l2_model['val_score'])
print('test accuracy: %s' % l2_test_accuracy)


# Run the following cell to produce the coefficients corresponding to 10 words in the dataset for the 'L2' regularization. You notice that some coefficients reduce close to zero, but never are completley eliminated from the regression. The array returned is set up such that the first column is the word, the second column is the index of the word, and the third column is the corresponding coefficient associated with the word.

# In[23]:


num_tokens = len(token2index)
keys = list(token2index.items())
keys = np.array(keys).reshape(len(token2index), 2)
l2_coefficients = np.append(keys, l2_model['best_model'].coef_.reshape(num_tokens, 1), 1)
l2_coefficients[14:25, :]


# In[24]:


l1_model = train_and_val_regularized(X_train, Y_train, X_val, Y_val, 'l1', 0.45) 
l1_test_accuracy = test_accuracy(l1_model['best_model'], X_test, Y_test)
print('train accuracy: %s' % l1_model['train_score'])
print('val accuracy: %s' % l1_model['val_score'])
print('test accuracy: %s' % l1_test_accuracy)


# Run the following cell to produce the coefficients corresponding to 10 words in the dataset for the 'L1' regularization. You notice that some coefficients reduce to zero. It may not be surprising that words such as 'amazing' are associated with a positive movie review, but it is interesting to look at the word 'good' as this is actually associated with a negative movie review.

# In[25]:


l1_coefficients = np.append(keys, l1_model['best_model'].coef_.reshape(num_tokens, 1), 1)
l1_coefficients[14:25, :]


# ### Plotting the Coefficients and Cross-validation Plots
# 
# Finally, we will plot the coefficient path and cross-validation vs. $C$ plots for both L1 and L2 regularization. Remember that $C=1/\lambda$, so a low $C$ actually corresponds to a higher $\lambda$. In addition, the plots are on a natural log scale in order to help with visualization.

# In[26]:


plot_coeffs_and_CV(X_train=X_train, Y_train=Y_train, cv=5, penalty='l1')


# In[27]:


plot_coeffs_and_CV(X_train=X_train, Y_train=Y_train, cv=5, penalty='l2')


# # Summary
# 
# In this , we have developed logistic regression model and applied regularization techniques to it. 

# In[ ]:




