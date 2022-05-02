#!/usr/bin/env python
# coding: utf-8

# In[37]:


from typing import Any, Tuple, Dict, Iterable, List

import numpy as np
import matplotlib.pyplot as plt

#from testing import TestHelper
from utils import *


# In[38]:


X_small = np.linspace(-5,5,15).reshape(15,1)
Y_small = np.array([-14,-3.5,-26,-6.25,23,5,5.25,17.5,1.3,8.7,17,23,-2,3,5]).reshape(15,1)


# In[39]:


from matplotlib import ticker, cm
SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.scatter(X_small, Y_small)
plt.xlabel('X_small')
plt.ylabel('Y_small', rotation=0, labelpad=30)
plt.show()


# #### Matrix Representation for Linear Regression
# 
# For a standard linear regression, we know that $\textbf{y}=\textbf{X}\boldsymbol{\theta}$. This is set up as a design matrix representation :
# 
# $$\begin{bmatrix}y_{1}\\y_{2}\\y_{3}\\\vdots\\y_{n}\end{bmatrix}=\begin{bmatrix}
#     1 & x_{11} & x_{12} & x_{13} & \dots  & x_{1m} \\
#     1 & x_{21} & x_{22} & x_{23} & \dots  & x_{2m} \\
#     \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
#     1 & x_{n1} & x_{n2} & x_{n3} & \dots  & x_{nm}
# \end{bmatrix}\begin{bmatrix}\theta_{0}\\\theta_{1}\\\theta_{2}\\\vdots\\\theta_{m}\end{bmatrix}$$
# 
# Run the following cell to create the design matrix (with the `design_matrix` function ). You will then consider the dimensionality of the parameter vector $\boldsymbol{\theta}$ . 

# In[40]:


X_small = design_matrix(X_small)
print(X_small)


# ###  Initializing the Parameter Vector
# Below,we will set up a function to initialize the parameter vector to all zeros for any size dataset.

# In[41]:


def initialize_parameters(X: np.ndarray) -> np.ndarray:
    """This function initializes the model parameters to all zeros

    Parameters
    ---------
    X: type `np.ndarray`, shape (N, M+1) 
        Design matrix, where N, M are the number of datapoints and number of 
        features in X, respectively

    Returns parameter vector theta as an np.ndarray of shape (?,1). It is you job to figure 
    out how many rows are in the array.
    """
    ### YOUR CODE HERE
    N,features = X.shape
    theta = np.zeros((features,1),dtype = float)
    return theta


# In[42]:


theta0 = initialize_parameters(X_small)
print(theta0) # should be [[0.], [0.]]


# In[43]:


initial_MSE = loss(X_small, Y_small, theta0) 
print('MSE = %0.4f' % initial_MSE)
plt.scatter(X_small[:,1], Y_small)
plt.plot(X_small[:,1], predict(X_small, theta0), c='k')
plt.xlabel('X_small')
plt.ylabel('Y_small', rotation=0, labelpad=30)
plt.show()


# ###  Implementing Closed-form Solution
# 
# Now we will explore the closed-form solution for an ordinary least squares (OLS) regression:
# 
# $$\boldsymbol\theta=(\textbf{X}^{T}\textbf{X})^{-1}\textbf{X}^{T}\textbf{Y}$$
# 
# 
# 
# 

# In[44]:


def closed_form_linear_regression(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """This function returns the closed-form solution to an ordinary least squares regression.

    Parameters
    ---------
    X: type `np.ndarray`, shape (N, M+1) 
        Design matrix, where N is the number of datapoints and M is the number of features.
    Y: type np.ndarray, shape (N, 1), 
        where N is the number of datapoints.
    
    Returns parameter vector of type np.ndarray, shape (M+1, 1) where M is the number of features.
    """ 
 
    theta = np.linalg.inv(X.T@X)
    theta = theta @(X.T@Y)
    return theta 


# ###  Solving For the Closed-form Solution
# 
# Below, you will use the function you just implemented above to solve for the optimized parameters $\boldsymbol{\theta}$ and the minimized MSE associated with these parameters for the small dataset (`X_small`, `Y_small`). 

# In[45]:


optimized_theta = closed_form_linear_regression(X_small, Y_small)


# In[46]:


print('MSE = %0.4f' % loss(X_small, Y_small, optimized_theta)) # should be MSE = 65.7408
print('The optimized parameters using the closed-form solution is: %s' % optimized_theta) # should be [[3.8], [1.97225]]
plt.scatter(X_small[:, 1], Y_small)
plt.plot(X_small[:, 1], predict(X_small, optimized_theta), c='k')
plt.xlabel('X_small')
plt.ylabel('Y_small', rotation=0, labelpad=30)
plt.show()


# ## Stochastic Gradient Descent
# 
# In this section, you will implement the stochastic gradient descent algorithm  You will then analyze the effects of small learning rates and large learning rates) on convergence.
# 
# 

# ### Implementing the Gradient
# Below, we will write the function `gradient` which computes the gradient for your SGD algorithm. Remember the function for the gradient in an OLS regression at any point $i$ is: 
# 
# $$\nabla J^{(i)}(\boldsymbol{\theta})=(\boldsymbol{\theta}^T\textbf{x}^{(i)}-y^{(i)})\textbf{x}^{(i)}$$ 
# 
# 

# In[47]:


def gradient(X: np.ndarray, Y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """This function returns the gradient of a point or points.

    Parameters
    ---------
    X: type `np.ndarray`, shape (N, M+1) 
        where N is the number of datapoints and M is the number of features.
    Y: type `np.ndarray`, shape (N, 1), 
        where N is the number of datapoints.
    theta: type `np.ndarray`, shape (M+1, 1)
        Model parameters
    
    Returns the gradient of type `np.ndarray`, shape (M+1, 1)
    """ 
    ### YOUR CODE HERE
    Y_pred = predict(X, theta)
    delta = ((Y_pred - Y)*X).T
    return delta  


# In[48]:


# Edit as desired! Remember to keep them as arrays
X = np.array([[1.0, 1.0, 0.0]]) 
Y = np.array([[1.0]])  
theta = np.array([[1.0], [2.0], [3.0]]) 

print('Your gradient is: \n \n %s.' % gradient(X, Y, theta)) 


# ###  Implementing Stochastic Gradient Descent
# 
# Implement the `train` function that takes in the training dataset `X_train, Y_train`, the initial parameters $\boldsymbol{\theta_0}$, number of epochs, learning rate, and returns the optimized parameter vector $\boldsymbol{\theta}$. 

# In[49]:


def train(X_train: np.ndarray, 
          Y_train: np.ndarray, 
          theta0: np.ndarray, 
          num_epochs: int, 
          lr: float) -> np.ndarray:
    """This function optimizes the parameters of linear regression through SGD.

    Parameters
    ---------
    X_train: type `np.ndarray`, shape (N, M+1)
        2D numpy array of training data, where N is the number of datapoints and 
        M is the number of features. 
    Y_train: type `np.ndarray`, shape (N, 1)
        1D numpy array of training label, where N is the number of datapoints
    theta0: type `np.ndarray` shape (M+1, 1)
        Initial model parameters for M features and plus 1 for bias column
    num_epochs: type `int`
        Number of training epochs
    lr: type `float`
        Learning rate
    
    Returns the updated parameter vector of type np.ndarray, shape (M+1, 1)
    """ 
  
    N,M = X_train.shape
    # for each epoch 
    for i in range(num_epochs):
        # for each training instance
        for j in range(len(X_train)):
            # extract and reshape X_i and y_i 
            X_j = X_train[j].reshape(1,M)
            Y_j = Y_train[j].reshape(1,1)
            
            # update theta
            theta0 = theta0 - lr * gradient(X_j,Y_j,theta0)
    
    return theta0 


# In[50]:


theta = train(X_train=X_small, Y_train=Y_small, theta0=theta0, num_epochs=500, lr=0.001)
print(theta) # should be [[3.76438958], [1.90920772]] for num_epochs=500 and lr=0.001


# ### Effects of Varying the Learning Rates on Convergence
# 
# In this question, we will analyze of varying the learning learning rates on the convergence of SGD, based on the figures below: 
# 
# ![image](img/varying_lr.png)
# 
# Here are the descriptions of the figure: 
# - The top left figure shows the loss vs. epoch graph when the learning rate is 0.1
# - The top right figure shows the loss vs. epoch graph when the learning rate is 0.01
# - The bottom left figure shows the loss vs. epoch graph when the learning rate is 0.001
# - The bottom right figure shows the loss vs. epoch graph when the learning rate is 0.0001
# 
# B

# ### Q: Does a decrease in learning rate require more or less epochs for convergence?
#  
#  ANS: A decrease in learning rate requires more epochs for convergence

# ### Q: What is happening with the largest learning rate, 0.1?
# 
# ANS:It is taking too large a step and overshooting the minimum of the objective function.

# In[51]:


lr = 0.01
visualize_SGD_loss(X=X_small, Y=Y_small, theta0=theta0, num_epochs=100, lr=lr, gradient_fn=gradient)


# ### Comparing Stochastic Gradient Descent with Gradient Descent
# 
# We will be using the [NASA 'airfoil' dataset](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise#) for analysis in this section. Mainly, we will be comparing the convergence behaviors of Stochastic Gradient Descent and Gradient Descent.
# 

# In[54]:


X_train, Y_train = load_data('./data/train.csv')
X_test, Y_test = load_data('./data/test.csv')
X_train = design_matrix(X_train)
X_test = design_matrix(X_test)


# In[55]:


import time
stime = time.time()
visualize_GD_loss(X=X_train, 
                  Y=Y_train, 
                  theta0=initialize_parameters(X_train), 
                  lr=0.001, 
                  num_epochs=2000, 
                  gradient_fn=GD_gradient)
print("Time for GD fitting: %.3f seconds" % (time.time() - stime))


# Now run the cell below to see how long the *stochastic gradient descent* optimization takes with 10 epochs and learning rate of 0.001 and visualize the convergence process. Our implementation takes 0.1-0.3 second on average

# In[56]:


import time
stime = time.time()
visualize_SGD_loss(X=X_train, 
                   Y=Y_train, 
                   theta0=initialize_parameters(X_train), 
                   lr=0.001, 
                   num_epochs=10, 
                   gradient_fn=gradient)
print("Time for SGD fitting: %.3f seconds" % (time.time() - stime))


# It appears that SGD is a bit faster compared to GD. However, an astute reader may ask: the number of epochs set for SGD (10) is much less than GD (2000), of course SGD is quickier! To answer that question, note that the *the amount of epochs* it takes for SGD to converge is *much less* than GD. If GD needs more than 2000 epochs to converge at learning rate of 0.001, then SGD only needs less than 10 epochs. 
# 
# To illustrate this point further, we will now place the plots of SGD and GD side-by-side below. Run the cell below, with learning rate of 0.001 and 500 epochs.

# In[57]:


lr = 0.001
num_epochs = 500
visualize_GD_loss(X=X_train, 
                  Y=Y_train, 
                  theta0=initialize_parameters(X_train), 
                  lr=lr, 
                  num_epochs=num_epochs, 
                  gradient_fn=GD_gradient)
visualize_SGD_loss(X=X_train, 
                  Y=Y_train, 
                  theta0=initialize_parameters(X_train), 
                  lr=lr, 
                  num_epochs=num_epochs, 
                  gradient_fn=GD_gradient)


# You can see clearly that SGD converges much faster than GD, as expected. You are encouraged to play around with the learning rate and epoch number of values for both GD and SGD and see if SGD still converges faster than GD (Hint: it should). For example, try increases the learning rate of GD to 0.1 (which should make GD converges faster as it is taking larger step), keep SGD learning rate at 0.001, and change number of epochs of both to 50 (for clearer view).

# You will now look at how well SGD predicts the outputs on the following parity plot.

# In[ ]:


theta = train(X_train=X_train,
              Y_train=Y_train,
              theta0=initialize_parameters(X_train),
              lr=0.001,
              num_epochs=100)
Y_train_pred = predict(X_train, theta)
Y_test_pred = predict(X_test, theta)

fig, ax = plt.subplots(1, 2, figsize = (20,6))
    
# Create contour plot of the true error
ax[0].set_title('Training Set', fontsize=BIGGER_SIZE) 
ax[0].plot([100,150],[100,150], color='blue', linewidth=3)
ax[0].scatter(Y_train, Y_train_pred, color='black')
ax[0].set_xlabel('Actual Sound Pressure')
ax[0].set_ylabel('Predicted Sound Pressure')

#Solution Line
ax[1].set_title('Test Set',fontsize=BIGGER_SIZE) 
ax[1].plot([100,150],[100,150],color='blue', linewidth=3)
ax[1].scatter(Y_test, Y_test_pred, color='black')
ax[1].set_xlabel('Actual Sound Pressure')
ax[1].set_ylabel('Predicted Sound Pressure')


# # Summary 
# 
# 
# - Implement the closed-form solution for the Ordinary Least Squares problem
# - Implement the stochastic gradient descent algorithm for linear regression
# - Analyze the effects of learning rates on the convergence of SGD 
# - Compare the convergence behaviors of gradient descent and stochastic gradient descent

# In[ ]:





# In[ ]:





# In[ ]:




