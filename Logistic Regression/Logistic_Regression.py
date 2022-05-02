#!/usr/bin/env python
# coding: utf-8

# In[2]:


from typing import Tuple, List, Dict, Any

import time

import numpy as np
import matplotlib.pyplot as plt

from testing import TestHelper
from utils import *


# The sample dataset is defined below. A few notes on the dataset: 
# * This dataset includes a training set of 4 instances and a validation set of 2 instances. 
# * The labels are binary (i.e. 0 or 1 predictions)
# * Each instance has a 2-element feature vector (e.g. `X_train[0] = [1.0, 1.0]`). However, the first element is a 
# *bias feature* (i.e. the feature associated with the bias term) and thus always takes the value of 1. This is a "notation trick"s to make computations more convenient. As a result, this dataset only has 1 actual feature.
# 
# The visualization of the training set `X_train, y_train` is shown in the plot below. The orange points and blue points are labeled `1` and `0`, respectively. The y-axis also denotes this labeling. The single feature is denoted in the x-axis.

# In[3]:


# setting up sample data
sample_X_train = np.array([[1.0, 1.0], 
                        [1.0, 2.0], 
                        [1.0, 4.0], 
                        [1.0, 5.0]])
sample_Y_train = np.array([[0], [0], [1], [1]])   
sample_X_val = np.array([[1.0, 1.5], 
                      [1.0, 6.0]])
sample_Y_val = np.array([[0], [1]])


# <img src="./assets/toy_dataset.png" width="500" />

# ###  Implement Objective Function 
# 
# Implement the function `objective` that computes the *average* negative log-likelihood (NLL) objective that we want to minimize. As a reminder, the average NLL is computed as follows:
# 
# \begin{split}
# J(\boldsymbol{\theta}) &= \frac{1}{N} \sum_{i=1}^N J^{(i)}(\boldsymbol{\theta}) \\
# &= - \frac{1}{N} \sum_{i=1}^N (y^{(i)}\log \mu^{(i)} + (1-y^{(i)})\log(1-\mu^{(i)})) \\
# \end{split}
# 
# where
# $$\mu^{(i)} = P(y^{(i)} = 1 \mid \mathbf{x}^{(i)}, \boldsymbol{\theta}) = \frac{1}{1 + e^{-\boldsymbol{\theta}^T \mathbf{x}^{(i)}}}$$
# 
# 

# In[6]:


def objective(X: np.ndarray, Y: np.ndarray, theta: np.ndarray) -> float:
    """Compute the average negative log-likelihood of the binary logistic regression model
    
    Parameters
    ----------
    X: type `np.ndarray`, shape (N, M+1)
        2D numpy array of data, with each row represents an instance (N instances in total) 
        and each column a feature (M features + 1 for bias term)
    Y: type `np.ndarray`, shape (N, 1)
        2D numpy array of label
    theta: type `np.ndarray` shape (M+1, 1)
        Model parameters
        
    Returns the average NLL
    """
    ### YOUR CODE HERE
    
    N,M = X.shape
    nll = 0
    for i in range(N):
        mu_i = (1/(1 + np.exp(-(theta.T @ X[i].reshape(M,1)))))
        nll += -(Y[i,0] *np.log(mu_i[0,0]) + (1 - Y[i,0])*np.log(1-mu_i[0,0]))
    return nll/N 


# In[8]:


theta = np.array([[0.0], [0.0]])
avg_nll = objective(sample_X_train, sample_Y_train, theta)
print(avg_nll) # should be 0.6931471805599453


# ###  Implement Gradients 
# 
# Implement the function `gradient` that computes the gradient of the per-example negative log-likelihood $\nabla_{\boldsymbol{\theta}} J^{(i)}$, given a single training data $\mathbf{x}^{(i)}, y^{(i)}$ and the model parameters $\boldsymbol{\theta}$. As a reminder, the per-example gradient is computed as 
# 
# $$\nabla_{\boldsymbol{\theta}} J^{(i)} = - (y^{(i)} - \mu^{(i)})\mathbf{x}^{(i)} \quad \text{where} \quad \mu^{(i)} = \frac{1}{1 + e^{-\boldsymbol{\theta}^T \mathbf{x}^{(i)}}}$$

# In[9]:


def gradient(X_i: np.ndarray, Y_i: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Compute the per-example gradient of the binary logistic regression model
    
    Parameters
    ----------
    X_i: type `np.ndarray`, shape (M+1, 1)
        The feature vector of example i, consists of M features + 1 for bias term
    Y_i: type `np.ndarray`, shape (1, 1)
        2D array containing the label of example i
    theta: type `np.ndarray`, shape (M+1, 1)
        Model parameters
        
    Returns per-example gradient, shape (M+1, 1)
    """
    mu_i = 1/(1 + np.exp((-(theta.T @ X_i))))
    delta_i = -(Y_i - mu_i)*X_i
    return delta_i 


# In[12]:


theta = np.array([[0.0], [0.0]])
num_features = sample_X_train.shape[1]

# get a single data point i
X_i = sample_X_train[0].reshape((num_features,1))
Y_i = sample_Y_train[0].reshape((1,1))

grad = gradient(X_i, Y_i, theta)
print(grad) 


# ###  Implement Stochastic Gradient Descent 
# 
# Implement the function `train` that takes in the training data, the intial parameters $\boldsymbol{\theta}_0$, number of epochs, learning rate, and trains our model using Stochastic Gradient Descent (SGD). 

# In[13]:


def train(X_train: np.ndarray, 
          Y_train: np.ndarray, 
          theta0: np.ndarray, 
          num_epochs: int, 
          lr: float) -> List[np.ndarray]:
    """Train logistic regression model with Stochastic Gradient Descent
    
    Parameters 
    ----------
    X_train: type `np.ndarray`, shape (N, M+1)
        2D numpy array of training data, 
        where N = number of examples and M = number of features
    Y_train: type `np.ndarray`, shape (N, 1)
        1D numpy array of training label
    theta0: type `np.ndarray` shape (M+1, 1)
        Initial model parameters
    num_epochs: type `int`
        Number of training epochs
    lr: type `float`
        Learning rate
    
    Returns a list of (num_epochs + 1) parameter vectors of type numpy array, starting with theta0 
    and then the parameter vector after each epoch. Each vector should have shape (M+1, 1)
    """    
    N, num_features = X_train.shape
    theta_history = [theta0]
    theta = theta0
    for _ in range(num_epochs): 
        for i in range(N):
            X_i = X_train[i].reshape((num_features, 1)) # convert from shape (M+1,) to shape (M+1, 1)
            Y_i = Y_train[i].reshape((1, 1))            # convert from shape (1,) to shape (1, 1)
            theta = theta - lr*gradient(X_i,Y_i,theta)                               # TODO please fill out 
        theta_history.append(theta)
    return theta_history


# In[15]:


theta0 = np.array([[0.0], [0.0]])
num_epochs = 3
lr = 0.5
theta_history = train(sample_X_train, sample_Y_train, theta0, num_epochs, lr)
print(theta_history)


# ###  Implement Predictions 
# 
# Finally, implement function `predict` that makes predictions $\hat{y}$ on a given $\mathbf{x}$ and learned parameters $\boldsymbol{\theta}$. As a reminder, if $P(y = 1 \mid \mathbf{x}, \theta) \geq 0.5$, then we assign $\hat{y} = 1$, and 0 otherwise. 
# 

# In[16]:


def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Make (binary) predictions from a logistic regression model with parameters theta, 
    for a given X

    Parameters 
    ----------
    X: type `np.ndarray`, shape (N, M+1)
        2D numpy array of data, where N = number of examples and M = number of features
    theta: type `np.ndarray` shape (M+1, 1)
        (Learned) model parameters
  
    Returns predicted label Y_hat, numpy array of shape (N, 1)
    """    
    ### YOUR CODE HERE
    N, M = X.shape
    Y = []
    for i in range (N):
        X_i = X[i].reshape((M,1))
        mu_i = 1/(1 + np.exp((-(theta.T @ X_i))))
       
        if mu_i[0,0] >= 0.5:
            Y_i = 1
        else:
            Y_i = 0
        Y.append([Y_i])
    return np.array(Y)


# In[18]:


theta = np.array([[0.0], [0.0]])
Y_hat_train = predict(sample_X_train, theta)
print(Y_hat_train) # should be [[1], [1], [1], [1]]


# ### Analysis I: Model Selection

# In[19]:


def train_and_val(X_train: np.ndarray, 
                  Y_train: np.ndarray, 
                  X_val: np.ndarray, 
                  Y_val: np.ndarray, 
                  theta0: np.ndarray, 
                  num_epochs: int, 
                  lr: float, 
                  visualize_nlls: bool = True
                  ) -> Dict[str, Any]:
    """Training and validation of binary logistic regression model using Stochastic Gradient Descent
    
    Parameters
    ----------
    X_train: type `np.ndarray`, shape (M+1, N_train)
        2D numpy array of training data, with each row represents an instance and each column a feature
    Y_train: type `np.ndarray`, shape (N, 1)
        2D numpy array of training label
    X_val: type `np.ndarray`, shape (M+1, N_val)
        2D numpy array of validation data
    Y_val: type `np.ndarray`, shape (N_val, 1)
        2D numpy array of validation label
    theta0: type `np.ndarray` shape (M+1, 1)
        Initial model parameters
    num_epochs: type `int`
        Number of training epochs
    lr: type `float`
        Learning rate
    
    Returns the dictionary containing useful information such as the best parameters (best_theta)
    """

    # train the model
    theta_history = train(X_train, Y_train, theta0, num_epochs, lr)

    # get the training and validation negative log likelihoods
    train_nlls, val_nlls = [], []
    best_train_nll, best_val_nll, best_theta, best_epoch = None, None, None, None
    for epoch, theta in enumerate(theta_history): 

        # compute nlls
        train_nll = objective(X_train, Y_train, theta)
        val_nll = objective(X_val, Y_val, theta)
        train_nlls.append(train_nll)
        val_nlls.append(val_nll)

        # get best parameters
        if best_val_nll is None or val_nll < best_val_nll:
            best_val_nll = val_nll
            best_train_nll = train_nll
            best_epoch = epoch
            best_theta = theta

    # Get error rate
    Y_train_hat = predict(X_train, best_theta)
    Y_val_hat = predict(X_val, best_theta)
    train_error = error_rate(Y_train, Y_train_hat)
    val_error = error_rate(Y_val, Y_val_hat)


    # NLL vs epochs visualization
    if visualize_nlls:
        num_epochs = [i for i in range(num_epochs + 1)]
        plot_nlls(num_epochs, train_nlls, val_nlls)

    return {
        'best_theta': best_theta,         # best parameters
        'best_epoch': best_epoch,         # epoch that produces the best parameters
        'best_train_nll': best_train_nll, # best (lowest) average train nll
        'best_val_nll': best_val_nll,     # best (lowest) average val nll
        'train_error': train_error,       # train error of the model with the best parameters
        'val_error': val_error,           # val error of the model with the best parameters
        'theta_history': theta_history    # list of parameters produced at each epoch
    }


# The reference plot below shows the average NLL versus epochs. The blue and orange curves represent the training and validation average NLLs, respectively. You should observe that the training curve goes down as training continues, indicating that model training is proceeding correctly.

# <img src="./assets/nlls.png" width="500" />

# Please run the cell below to see your output and graph. If you implemented questions 1-4 correctly, your graph should look exactly the same as the reference graph above. In addition, our reference output is: 
# 
# | Output | Reference value |
# | :----------- | :----------- |
# | Best parameters | [[-3.88513022], [1.78311662]] |
# | Best epoch | 20 | 
# | Train NLL | 0.1766  |
# | Train Error Rate | 0.0000 (100% accuracy) |
# | Val NLL | 0.1310  |
# | Val Error Rate | 0.0000 (100% accuracy) |

# In[23]:


# setting up the experimental parameters
num_features = sample_X_train.shape[1] # NOTE: this includes the bias
theta0 = np.array([[4.0], [4.5]])
num_epochs = 20
lr = 0.5

# training and validation
output = train_and_val(X_train=sample_X_train, 
                       Y_train=sample_Y_train, 
                       X_val=sample_X_val, 
                       Y_val=sample_Y_val, 
                       theta0=theta0, 
                       num_epochs=num_epochs, 
                       lr=lr, 
                       visualize_nlls=True)

# print outputs
print('The best parameters theta = %s are found at epoch %d, with train nll = %0.4f, val nll = %0.4f, train error_rate = %0.4f, val error_rate = %0.4f' % (output['best_theta'], output['best_epoch'], output['best_train_nll'], output['best_val_nll'], output['train_error'], output['val_error']))


# ### Analysis II: Optimization Path and Decision Boundary Visualization
# 
# So far, you have implemented and trained a logistic regression model using SGD. Ultimately, what does this all mean? In this section, you will try to answer the question above with two visualizations like the ones shown below. These reference visualizations illustrate the behavior of  model when training correctly on the sample dataset.
# 
# The plot on the left denotes the *optimization path of SGD on the contour plot*. We hope you are a bit familiar with this plot now as you have seen it in the lecture videos as well as the M4 and M5 programming assignments. A few notes on the plot: 
# * The axes denote the values of parameters $\boldsymbol{\theta} = [\theta_1, \theta_2]^T$, where $\theta_1$ the bias term and $\theta_2$ is the weight of the only feature $x$ in our toy dataset. 
# * The numeric labels on the contour denote the values of the average NLL objective that we use to optimize with SGD.
# * The red arrows denote the path SGD took at each epoch. You can see that SGD took bigger steps at the beginning and then smaller one as we get closer to the local minimum. Reflect for a moment on why you think this occurs. 
# 
# The plot on the right denotes the behavior of the trained binary logistic regression model on the toy training data. Specifically, it shows the *decision boundary* (the red line) and the *logistic curve* (black curve) used to compute the probability $P(y=1|\boldsymbol{\theta}, \mathbf{x})$: 
# * We observe that the logistic curve fits our training data well, with data points labeled $y=1$ are assigned probability values greater than 0.5 (in fact very closed to 1). Similarly, the data points labeled $y=0$ are assigned probability values less than 0.5. 
# * The corresponding decision boundary splits the feature plane (or line, as there is only a single feature in our toy dataset) into two sections:  the points to the left of the decision boundary will be labeled $y=0$, and the points to the right will be labeled $y=1$.

# In[25]:


plot_sgd(sample_X_train, sample_Y_train, objective, output['theta_history'])
plot_decision_boundary(sample_X_train, sample_Y_train, output['best_theta'], xlim=(0,6), ylim=(-1,2))


# #  Binary Logistic Regression for Natural Language Processing
# 
# Now that we have implemented binary logistic regression, let's apply it to predict whether a movie review is negative or positive! In this section, you will learn how to convert a Natural Language Processing (NLP) dataset, namely the [Movie Review Polarity dataset](https://www.kaggle.com/nltkdata/movie-review), into the appropriate representation that this logistic regression model can learn. You will then train the model on the dataset and make predictions.

# 
# ### Dataset: Movie Review Polarity
# 
# The Movie Review Polarity dataset consists of labeled positive and negative movie reviews. 
# - `debug` consists of two example reviews, one for positive and one for negative. You will use this dataset for debugging purposes.
# - `full` contains the all of the movie reviews, which are already split into train, validation, and test set. We also shuffled the training dataset for you.
# 
# Run the code cell below to define the filepaths of the dataset as well as the constant `BIAS` token, which is explained later.

# In[33]:


# debug dataset
DEBUG_FILEPATH = './data/debug/reviews.tsv'
DEBUG_DICT_FILEPATH = './data/debug/dict.txt'

# full dataset
TRAIN_FILEPATH = './data/full/train_data.tsv'
VAL_FILEPATH = './data/full/valid_data.tsv'
TEST_FILEPATH = './data/full/test_data.tsv'
FULL_DICT_FILEPATH = './data/full/dict.txt' 

# token for bias term
BIAS = '<BIAS>'


# In terms of specific format, each line in the data file represents an example. Each example contains a movie review and a label, which is denoted 1 for positive and 0 for negative. The label is the first character of the line and is separated from the review using tab (`\t` character). The reviews consist of lowercase words and punctuation separated using white-space. Note that this lowercasing and white-space separation are parts of the preprocessing of the dataset that were done for you. In reality, datasets are not as clean and you would usually need to do some preprocessing beforehand. In summary, the  format  of  each  data  point  (each  line)  is
# 
# `label\tword_1 word_2 word_3 ... word_k\n`
# 
# Run the following code to see the example in the `debug` file.

# In[34]:


with open(DEBUG_FILEPATH, 'r') as f:
    for line in f.readlines():
        print(line)


# ### Binary Bag-of-Word Model
# 
# Given the reviews above, you need to convert the words into a representation that our model can process. One of the most basic NLP representations is *bag-of-words (BOW)*. In this format, each data example (i.e. review) is represented by the word appearance while disregarding other linguistic information such as word order and other syntactic information (hence, "bag" of words). 
# 
# In more detail, we are often provided with a dictionary mapping all tokens (i.e. words) in the dataset into their corresponding indexes, which we denote `token2index`. Suppose there are $M$ words in this dictionary. Each word will correspond to a binary feature of whether the word appears in the review or not. Thus, the feature vector of an example will consist of $M$ binary elements. Since each word $w_i$ is mapped to index $i$ in the dictionary `token2index`, the element $i$ of the feature vector is set to 1 if the word $w_i$ appears in the review, and 0 otherwise. 
# 
# As a concrete example, supposed we have: 
# - A sentence `s = "Machine learning is fun"`
# - A dictionary `token2index = {'hello': 0, 'machine': 1, 'fun':2, 'good': 3, 'learning': 4, 'bad': 5, 'is': 6}` 
# 
# The feature vector of the sentence `s` is then `[0, 1, 1, 0, 1, 0, 1]`. This is because 'machine' (index 1), 'learning' (index 4), 'is' (index 6), and 'fun' (index 2) appear in s (thus assigned 1), and 'hello', 'good', 'bad' don't appear in s (and thus are assigned 0). 
# 
# Another important detail we need to address when using a logistic regression model for NLP tasks is the handling of the *bias term*. So far, we have been handling the bias term by prepending 1 as the first element of the feature vector (see [description of toy dataset](#LRonToy)). We could do so in this section. However, we think this creates a little inconvenience that can be a source of unnecessary bugs when you implement this section. *Specifically, we opt to append 1 at the end of the feature vector, as opposed to the beginning.* If you prepend 1 in the beginning, then this bias feature will have index 0, and you will then need to "plus 1" when handling indexes of the words as the first word will also have index 0 in the `token2index` dictionary. If we add 1 at the end, then the bias feature will instead have the last index in the dictionary and you won't have to "plus 1". As an example of this, the `token2index` dictionary will be `token2index = {'hello': 0, 'machine': 1, 'fun':2, 'good': 3, 'learning': 4, 'bad': 5, 'is': 6, '<BIAS>: 7}`, with the special `<BIAS>` token mapped to the last index. The feature vector of the sentence `s` is then `[0, 1, 1, 0, 1, 0, 1, 1]`, with the last element reserved for the bias feature.
# 

# In[35]:


def load_dict(dict_filepath: str) -> Dict[str, int]:
    """Read the dictionary from filepath into a Python dictionary
    
    Parameters
    ----------
    dict_filepath: type `str`
        Filepath of the dictionary mapping tokens to numerical index
        
    Returns a Python dictionary mapping tokens to numerical index
    """
    token2index = dict()

    # adding everything from dict.txt
    with open(dict_filepath, 'r') as f:
        for line in f.readlines():
            word, index = line.strip().split(' ')
            token2index[word] = int(index)

    # adding bias term
    token2index[BIAS] = len(token2index)
    return token2index


# In[36]:


token2index_debug = load_dict(DEBUG_DICT_FILEPATH)
print(token2index_debug)


# In[37]:


def load_data(filepath: str, token2index: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Read in the data with N examples and M features from filepath into the appropriate numpy arrays
    
    Parameters
    ----------
    filepath: type `str`
        Text file containing the data
    token2index: type `str`
        Dictionary mapping tokens to the corresponding indexes. This already includes the bias feature, thus
        the size of this dictionary is M+1
    
    Returns the tuple (X, Y), where X has shape (N, M+1), and Y has shape (N, 1)
    """
    ### YOUR CODE HERE
    # Pseudocode hints 
    
    # initialize X, Y lists 
    X, Y = [],[]
    # get the number of features and the bias index 
    num_features = len(token2index)
    # read in the filepath 
    bias_index = token2index[BIAS]
    with open(filepath,'r') as f:
        # for each line in the file 
        for line in f.readlines():
        
            # separate line into label and review, after stripping all the whitespaces 
            label, review = line.strip().split('\t')
            # initialize a numpy feature vector with all zeros
            X_i = np.zeros(num_features)
            # set the bias feature to 1 
            X_i[bias_index] = 1
            # for each word in the review
            for word in review.split(' '): 
            
                # if the word is in the vocab dictionary, 
                if word in token2index:
                    
                # set the feature associated with that word to 1
                    X_i[token2index[word]] = 1
                
            # add the feature vector and label to their respective X, Y lists
            X.append(X_i)
            Y.append([int(label)])
    # return numpy versions of X, Y
    return (np.array(X),np.array(Y))


# In[38]:


token2index_debug = load_dict(DEBUG_DICT_FILEPATH)
X_debug, Y_debug = load_data(DEBUG_FILEPATH, token2index_debug)
print('X=', X_debug)
print('Y=',Y_debug)


# ### Training and Evaluation
# 
# Now that you have implemented `load_data`, let's see how we can train our logistic regression model on this dataset. The following code block is almost identical to what you saw in [Part I Model Selection Analysis](#analysisI), except we now apply it to the Movie Review Polarity dataset. A few notes:
# - As this dataset is much bigger than the toy dataset. It is going to take a bit longer to train. We also output the time in seconds for you to see how long training takes. Our reference implementation takes 4-8 seconds on average.
# - We plot the average NLL vs. epochs for you to monitor the training process
# - If you implemented everything correctly, your output and plot should be the same as the reference output and plot below: 
# 
# | Output | Value |
# | :----------- | :----------- |
# | Best epoch | 7 | 
# | Train NLL | 0.0429  |
# | Train Error Rate | 0.0000 (100% accuracy) |
# | Val NLL | 0.3109  |
# | Val Error Rate | 0.1300 (87% accuracy) |
# | Test Error Rate | 0.1525 (84.75% accuracy) |
# 
# <img src="./assets/large_nlls.png" width="500" />
# 

# In[40]:


#UNCOMMENT TO TRAIN
# loading data
print('Loading data...', end='')
token2index = load_dict(FULL_DICT_FILEPATH)
X_train, Y_train = load_data(TRAIN_FILEPATH, token2index)
X_val, Y_val = load_data(VAL_FILEPATH, token2index)
X_test, Y_test = load_data(TEST_FILEPATH, token2index)
print('finished!')

# setting up the experimental parameters
num_features = len(token2index) # NOTE: this includes the bias
theta0 = np.zeros((num_features, 1)) # initialize parameters to all 0
num_epochs = 25
lr = 0.01

# training and validation
print('Start training...', end='')
start = time.time()
output = train_and_val(X_train, Y_train, X_val, Y_val, theta0, num_epochs, lr, visualize_nlls=True)
end = time.time()
print('finished!')

# finally testing
Y_hat_test = predict(X_test, output['best_theta'])
test_error = error_rate(Y_test, Y_hat_test)

# print outputs
print('The best parameters are found at epoch %d, with train nll = %0.4f, val nll = %0.4f, train error_rate = %0.4f, best val error_rate = %0.4f, test error_rate = %0.4f' % (output['best_epoch'], output['best_train_nll'], output['best_val_nll'], output['train_error'], output['val_error'], test_error))
print('Total time = %0.4fs' % (end-start))


# ### Making Predictions
# 
# After training and validation, we would like to see the result that our model would predict, given a movie review. In this code cell below, we show you an example of a movie review, which is the first line in the test dataset. A human reading the review would predict a negative (0) label. We then make predictions on that movie review to see whether our model predicts the review as positive or negative. If you implement everything correctly, the model will also predict the negative label after you run the code cell below (`True sentiment = 0`)
# 
# In general, you can use the `predict` function to predict any movie review, provided you convert the review into the [appropriate representation](#bow)
# 

# In[41]:


#UNCOMMENT TO MAKE PREDICTIONS
# print out the first review of the test dataset and its corresponding true sentiment
with open(TEST_FILEPATH, 'r') as f:
    first_line = f.readlines()[0]
    label, review = first_line.strip().split('\t')
    print('Review =', review)
    print('True sentiment =', label)

# make prediction. Because predict returns a 2D array, we index to get the element
review = predict(np.array([X_test[0]]), output['best_theta'])[0, 0]

# output
print('Predicted sentiment =', review)


# ## Summary
# 
# 
# - Build a binary logistic regression model
# - Train, evaluate, and visualize the behaviour of logistic regression when training with Stochastic Gradient Descent
# - Process a Natural Language Processing data (Movie Review Polarity) into the appropriate representation (binary Bag-of-Words) for learning
# - Train the logistic regression model and make predictions on the Movie Review Polarity
# 
# 

# In[ ]:





# In[ ]:




