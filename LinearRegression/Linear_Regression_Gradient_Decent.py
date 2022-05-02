#!/usr/bin/env python
# coding: utf-8

# <a name="data"></a>
# 
# ### Dataset Description
# The [NASA 'airfoil' dataset](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise#) was "obtained from a series of aerodynamic and acoustic tests of two and three-dimensional airfoil blade sections conducted in an anechoic [soundproof] wind tunnel." - R. Lopez, PhD
# 
# An airfoil is the cross-sectional shape of a wing, blade or propeller. For this particular dataset, specific variables were measured that are known to contribute to the amount of noise the airfoil generates when exposed to smooth air flow. As you can imagine, it's advantageous to design airfoils that are less noisy for their intended use. NASA specifically conducted these studies to predict noise production using the variables as follows. For more information about the prediction of noise production from airfoils, check out this [link](https://ntrs.nasa.gov/citations/19890016302).
# 
# Inputs: 
# - Frequency, in Hz
# - Angle of attack, in degrees
# - Chord length [length of airfoil], in meters
# - Free-stream velocity, in meters per second
# - Suction side displacement thickness, in meters
# 
# Output: 
# - Scaled sound pressure level, in decibels.
# 
# Preview:
# 
# | Pressure (dB) | Frequency (Hz) | Angle (degrees) | Length  (m) | Velocity (m/s) | Displacement (m) 
# |--|--|--|--|--|--|
# | 126.201 | 800 | 0 | 0.3048 | 71.3 | 0.002663 
# | 125.201 | 1000| 0 | 0.3048 | 71.3 | 0.002663 
# | 125.951 | 1250| 0 | 0.3048 | 71.3 | 0.002663 
# 
# We have preprocessed the data by standardizing the features so their mean is set to zero and their variance is set to one. This prevents issues of numerical overflow when optimizing with Gradient Descent, which can occur if a feature contains very large or very small values. The output variable, Pressure, is kept in the same units, dB. 

# In[84]:


from typing import List, Set, Dict, Tuple, Any

import numpy as np
import pandas as pd

from testing import TestHelper
from utils import * 

get_ipython().run_line_magic('matplotlib', 'inline')


TRAIN_FILE = './data/train.csv'   # filepath of full_train dataset
TEST_FILE = './data/test.csv'     # filepath of full_test dataset

VIZ_DATA_FILE = './data/viz_data.csv'     # filepath for visualization dataset


# # Loading data 

# In[85]:


def load_data(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load in the datafile

    Parameters
    ----------
    filename: type `str`
        filepath of the datafile 

    Returns X array of shape (M, N) and Y array of shape (N, 1)
    """    
  
    df = pd.read_csv(filename)
    Y = df['Pressure (dB)']
    Y = np.array(Y)
    X = df[['Frequency (kHz)','Angle (degrees)','Length (m)','Velocity (m/s)', 'Displacement (m)']]
    X = np.array(X)
    a = Y.shape
    Y = Y.reshape(a[0],1)
    return (X,Y)


# ### Creating the Design Matrix 
# 
# For this question, we will write a function called `design_matrix` which will take a NumPy array `X` and fold in a bias term by adding a column of ones to the front of `X`. This will allow you to use linear algebra operations for all your functions moving forward without worrying about having to separately handle the bias term at each step. As you'll see, the bias term will become another "weight" in your parameter vector $\theta$ that is learned during gradient descent.

# In[86]:


def design_matrix(X: np.ndarray) -> np.ndarray:
    """This function takes in X and creates a design matrix by 
    adding a column of ones to front of the array

    Parameters
    ----------
    X: type `np.ndarray`, shape (N, M)
        where M = number of features, N = number of observations

    Returns design matrix of type np.ndarray and shape (N, M+1)
    """
    X = np.concatenate([np.ones((X.shape[0],1),dtype=X.dtype),X], axis=1)
    return X 


# In[87]:


# Run this code to test your design_matrix function
X = np.array([[1.0, 3.0],
              [2.0, 2.0],
              [3.0, 1.0]]) # X shape = (3, 2) 
X = design_matrix(X)

print('Your design matrix is:')
print(X)


# ### Implementing Mean Squared Error 
# 
# For this question, we will write a function called `loss` that will calculate the mean squared error (MSE) between your true y-values and the predicted values generated from your linear equation. 
# 
# Remember that MSE $J(\boldsymbol{\theta})$ is defined in this course as:
# 
# \begin{split}
# J(\boldsymbol{\theta}) &= \frac{1}{N} \sum_{i=1}^{N} \frac{1}{2}(y_i - \hat{y_i})^2 \\
# &= \frac{1}{N} \sum_{i=1}^{N} \frac{1}{2}(y_i - \boldsymbol{\theta}^T \textbf{x}^{(i)})^2 \\
# \end{split}

# In[88]:


def loss(X: np.ndarray, Y: np.ndarray, theta: np.ndarray) -> float:
    """Calculate mean-squared error between the true y-values and the predicted y-values

    Parameters
    ----------
    X: type `np.ndarray`, (N, M+1)
        2D design matrix, N = number of observations, M = number of features
    Y: type `np.ndarray`, (N, 1)
        2D array of target true y-values
    theta: type `np.ndarray`, shape (M+1, 1)
        2D array of model parameters (weights + bias)

    Returns mean-squared error
    """
 
    N = X.shape
    Y_hat = X@theta
    MSE = np.square(Y-Y_hat).mean()
    MSE = MSE/2
   
    return MSE 


# In[89]:


# Create your own datasets to test/debug your loss function
X = np.array([[1.0, 0.0], 
              [1.0, 0.0], 
              [1.0, 0.0], 
              [1.0, 0.0]], ndmin = 2) 
theta = np.array([[0.0],
                  [0.0]], ndmin = 2)
Y = np.array([[1.0], 
              [1.0], 
              [1.0], 
              [1.0]], ndmin = 2) 

print('Your calculated mean squared error is %s.' % loss(X, Y, theta)) #  MSE should be 0.5


# #### Visualizing the mean-squared error loss function
# 
# We don't yet know exactly what parameters will allow us to minimize our loss function (mean-squared error between predictions and target y-values), but we can build some intuition for this by visualizing how our error changes with parameter values. Let's imagine we have a model defined by only one feature (with parameter $w_0$) and one intercept ($w_1$). We can plot our objective function for a range of these values and see how it changes. Below are the two reference plots: contour plot of the loss function (left), and the line of best fit (right). The point at the minimum of the loss function is where we want our solution to converge to.
# 
# ![image](img/loss_and_fit.png)
# 
# Run the cell below (and uncomment the `visualize_loss` line) to call the function `visualize_loss` on a randomly generated dataset with an elliptical objective (loss) function. The `visualize_loss` function takes in `X`, `Y`, and the `loss` function. 

# In[ ]:





# ###  Implementing Gradient 
# 
# Below, we will implement the function `gradient` which computes the linear regression gradient for our Gradient Descent algorithm. As a reminder, this gradient is computed as: 
# $$\nabla J(\boldsymbol{\theta}) = - \frac{1}{N} \sum_{i=1}^N(y^{(i)} - \boldsymbol{\theta}^T\textbf{x}^{(i)}) \textbf{x}^{(i)}$$ 

# In[90]:


def gradient(X: np.ndarray, Y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """This function calculates the gradient of the loss function with respect to theta

    Parameters
    ---------
    X: type `np.ndarray`, shape (N, M+1) 
        Design matrix, M = number of features, N = number of instances    
    Y: type `np.ndarray`, shape (N, 1), 
        Matrix of target values, N = number of instances
    theta: type `np.ndarray`, shape (M+1, 1)
        Model parameters, M = number of features
    
    Returns a np.ndarray of gradient for each parameter in theta, shape (M+1, 1) 
    """

    N, num_features = X.shape
    Y_hat = X@theta # shape (N, 1)
    gradients = (-1/N)*np.sum((Y - Y_hat)*X, axis=0) # shape (M+1,)
  
    gradients = np.reshape(gradients, (num_features, 1)) # shape (M+1,1)
    
    return gradients
    


# In[91]:


# Edit as desired! Remember to keep them as arrays
X = np.array([[1.0, 0.0, 0.0], # 4x3, four instances, 2 features + 1 constant column
              [1.0, 0.0, 0.0],
              [1.0, 1.0, 0.0],
              [1.0, 1.0, 0.0]]) 
Y = np.array([[1.0], 
              [1.0], 
              [0.0], 
              [0.0]], ndmin= 2)  # 4x1, 4 instances
theta = np.array([[0.0], 
                  [0.0], 
                  [0.0]], ndmin = 2) # 3x1, 3 weights

print('Your gradient is: \n \n %s.' % gradient(X,Y,theta)) 
# Correct answer is : 
# [[-0.5.]
#  [-0.]
#  [-0.]].


# #### Visualizing a single optimization step
# 
# Here we will visualize a single step of gradient descent. We show you the contour plots of the loss function (left) and the linear fits (right) below. 
# - On the left plot, you can see a single step from "initial weights" to "weights after one epoch." Note the dot representing the minimum point on the far left.
# - On the right plot, you can see the linear fits of "initial weights" and "weights after one epoch," with the corresponding names "initial fit" andd "fit after one epoch", respectively.
# 
# ![image](img/one_step_update.png)
# 
# 

# In[ ]:





# ###  Implementing Gradient Update 
# 
# Next we will implement the `update` function, which takes in the current parameter vector $\boldsymbol{\theta}$, the gradient `gradients` computed from `gradient`, the learning rate `lr`, and output the updated parameter vector

# In[63]:


def update(theta: np.ndarray, gradients: np.ndarray, lr: float) -> np.ndarray:
    """Update the model parameters

    Parameters
    ----------
    theta: type `np.ndarray`, shape (M+1, 1) 
        Previous parameters vector
    gradients: type `np.ndarray`, shape (M+1, 1) 
        Gradient vector
    lr: type `float`
        Learning rate 

    Returns the new theta, (M+1, 1) vector
    """

    return theta - lr*gradients 


# In[64]:


# Create your own datasets to test/debug your update
lr = 0.01 # set learning rate
theta = np.array([[0.0], 
                  [0.0], 
                  [0.0],
                  [0.0]], ndmin = 2) # shape (4, 1), 3 weights
gradients = np.array([[0.0], 
                      [0.0], 
                      [-1.0],
                      [-1.0]], ndmin = 2) # shape (4, 1), 3 weights

print('Your new theta vector is: ') 
print(update(theta, gradients, lr))

# Correct output is:

# [[0.  ]
#  [0.  ]
#  [0.01]
#  [0.01]]


# #### Visualizing Gradient Descent
# 
# Here we show you the optimization path of gradient descent. We show the reference plots below: 
# - The left plot indicates the contour plot for the objective function as well as the optimization path of gradient descent. 
# - The right plot shows how our line of best fit changes with each gradient descent time step (yellow means the first epoch; purple is the last epoch)
# 
# ![image](img/GD.png)
# 
# 

# In[ ]:





# ### Training your linear regression 
# 
# Now that we have each component of our algorithm written, our `loss`, `gradient`, and `update`, let's train our model by performing gradient descent for a set number of `num_epochs` to minimize the least-squares error.

# In[65]:


def train(X_train: np.ndarray, 
          Y_train: np.ndarray, 
          theta0: np.ndarray, 
          num_epochs: int, 
          lr: float) -> np.ndarray:
    """Train your linear regression with Gradient Descent

    Parameters
    ---------
    X_train: type `np.ndarray`, shape (N, M+1)
        Design matrix, M = number of features, N = number of observations
    Y_train: type `np.ndarray`, shape (N, 1)
        Matrix of target values, N = number of observations
    theta0: type `np.ndarray` shape (M+1, 1)
        Initial model parameters for M features, plus 1 for bias column
    num_epochs: type `int`
        Number of training epochs to run gradient descent for
    lr: type `float`
        Learning rate
    
    Returns an array of trained parameters theta, shape (M+1, 1)
    """ 
    theta = theta0
    for i in range(num_epochs):
        grad = gradient(X_train, Y_train, theta)
        theta = update(theta, grad, lr)    
    return theta

   


# In[66]:


##Use this cell to test your train function
X = np.array([[1.0, 0.0, 1.0], #4x3, four observations, 2 features + 1 constant column
              [1.0, 0.0, 1.0],
              [1.0, 1.0, 0.0],
              [1.0, 1.0, 0.0]])
Y = np.array([[1.0], 
              [1.0], 
              [0.0], 
              [0.0]], ndmin= 2)  #4x1, 4 observations
theta0 = np.array([[0.0], 
                   [0.0], 
                   [0.0]], ndmin = 2) #3x1, 3 weights
lr = 0.01
num_epochs = 2
final = train(X, Y, theta0, num_epochs, lr)
print('Your trained theta vector is:')
print(final)


# ###  Predicting with trained linear regression 
# 
#  Function to perform predictions on a test dataset using your trained linear regression. 
# 

# In[67]:


def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray: 
    """Make prediction with linear regression

    Parameters
    ----------
    X: type `np.ndarray`, shape (N, M+1) 
        Design matrix, M = number of features + 1, N = number of observations
    theta: type `np.ndarray`, (M+1, 1) 
        Trained model parameters, M = number of features + 1
    
    Returns Y_hat, an array of predicted y-values with shape (N, 1)
    """
    return X@theta


# In[68]:


# Use this cell to test your predict function
X = np.array([[1.0, 0.0, 0.0], # shape (4, 3), four observations, 2 features + 1 constant column
              [1.0, 0.0, 0.0],
              [1.0, 1.0, 0.0],
              [1.0, 1.0, 0.0]]) 
theta = np.array([[1.0], 
                  [2.0], 
                  [3.0]], ndmin = 2) # shape (3, 1), 3 weights
Y_hat = predict(X, theta)
print('Your predicted Y-values are:')
print(Y_hat)


# #### Visualizing Predictions
# 
# We visualize the predictions made by the trained linear regression model. We show the reference graph below. The x-axis shows the predicted `Y_hat` values, and the y-axis shows the true `Y` values. Try out different learning rates and epochs to see if you can get your predictions closer to the unity line (orange). 
# 
# ![image](img/predictions.png)
# 
# 

# In[ ]:





# ###  Final Product - Train and Test! 
# 
# Create a final function `train_and_test` that takes in a `train_filename`, `test_filename`, `num_epochs`, learning rate `lr`, and does the following:
# - Load train and test data from files with `load_data`
# - Create design matrices with `design_matrix`
# - Initialize $\boldsymbol{\theta}$ to all zeros
# - Train a new linear regression model with `train`
# - Predict outcomes of train and test datasets with `predict`
# - Compute train and test error rates with `loss`
# - Return the trained parameters $\boldsymbol{\theta}$, train error, and test error in an output dictionary 

# In[69]:


def train_and_test(train_filename: str, test_filename: str, num_epochs: int, lr: float) -> Dict[str, Any]: 
    """train_and_test function takes in a train and test dataset,
     performs predictions on train and test, 
     and returns train error and test error as a Tuple

    Parameters
    ----------
    train_filename: type `str`
        The filepath to the training file
    test_filename: type `str`
        The filepath to the testing file
    num_epochs: type 'int'
        Number of epochs to run gradient descent
    lr: type `float`
        Learning rate
    
    Returns the tuple of (train error rate, test error rate)
    """
   # load data
    X_train, Y_train = load_data(train_filename)
    X_test, Y_test = load_data(test_filename)
    
    # create design matrices
    design_X_train = design_matrix(X_train)
    design_X_test = design_matrix(X_test)
    
    # initialize parameters
    theta0 = np.zeros((design_X_train.shape[1], 1))
    
    # train linear regression
    theta_final = train(design_X_train, Y_train, theta0, num_epochs, lr)
    
    # predict labels for train and test
    train_predict = predict(design_X_train, theta_final)
    test_predict = predict(design_X_test, theta_final)

    # get the train error
    train_error = loss(design_X_train, Y_train, theta_final)
    test_error = loss(design_X_test, Y_test, theta_final)

    return {
        'theta'      : theta_final,
        'train_error': train_error, 
        'test_error' : test_error
    }


# In[55]:


# Use this cell to test your train_and_test function with the toy and full datasets
lr = 0.01 #set learning rate
num_epochs = 1000 #set number of epochs


full_output = train_and_test(TRAIN_FILE, TEST_FILE, num_epochs, lr)
print('Your train error on the full dataset is: %s' % full_output['train_error']) # should be 11.68
print('Your test error on the full dataset is: %s' % full_output['test_error']) # should be 11.02


# # Summary 
# 
# In this module, you learned how to build and train a linear regression model. Specifically, you learned how to: 
# - Implement a function to create a design matrix $\textbf{X}$
# - Implement the mean squared error loss function and visualize the contour plots
# - Implement gradient descent optimization and visualize the optimization path as it converges to a solution
# - Train a linear regression model and use it to make predictions
# 

# In[ ]:





# In[ ]:




