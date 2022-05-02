from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(filename, header = 0)
    data = np.asarray(data)
    X = data[:,1:]
    Y = data[:,0].reshape(len(data[:,0]),1) #Nx1
    
    return X, Y

def design_matrix(X: np.ndarray) -> np.ndarray:
    """This function takes in a data array X and creates a design matrix by 
    adding a column of ones to first column of the array

    Parameters
    ----------
    X: type `np.ndarray`, shape (N, M)
        where M = number of features, N = number of observations

    Returns design matrix of type np.ndarray and shape (N, M+1)
    """
    new_column = np.ones((X.shape[0], 1))
    design_matrix = np.concatenate((new_column, X), axis = 1)
    return design_matrix

def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray: 
    """Make prediction with linear regression

    Parameters
    ----------
    X: type `np.ndarray`, shape (N, M+1) 
        Design matrix, M = number of features + 1, N = number of observations
    theta: type `np.ndarray`, (M+1, 1) 
        Trained model parameters, M = number of features + 1
    
    Returns Y_pred, an array of predicted y-values with shape (N, 1)
    """    
    return X@theta

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
    Y_hat = X@theta
    mse = np.square(Y - Y_hat).mean()

    return mse / 2

def visualize_lrs(X: np.ndarray, 
                  Y: np.ndarray, 
                  theta0: np.ndarray, 
                  lrs: List[float], 
                  num_epochs: int,
                  gradient_fn, 
                  save_fig: bool = True) -> np.ndarray:
    
    fig, ax = plt.subplots(2, 2, figsize=(20,12)) #2x2 grid of plots
    
    for j, lr in enumerate(lrs):
        N, num_features = X.shape
        losses = [loss(X, Y, theta0)]
        theta = theta0
        for _ in range(num_epochs):
            for i in range(N):
                X_i = X[i].reshape((1, num_features))
                Y_i = Y[i].reshape((1, 1))
                theta = theta - lr*gradient_fn(X_i, Y_i, theta)
            losses.append(loss(X, Y, theta))
        losses = np.asarray(losses)
        ax[j//2, j%2].plot(np.arange(len(losses)), losses, label='SGD for lr=%s' % lr)
        ax[j//2, j%2].set_xlabel("Epochs")
        ax[j//2, j%2].set_ylabel("Loss", rotation=0, labelpad=25)
        ax[j//2, j%2].legend(loc='upper right')
        
    if save_fig: 
        fig.savefig('img/varying_lr.png')
        
def visualize_SGD_loss(X: np.ndarray, 
                       Y: np.ndarray, 
                       theta0: np.ndarray, 
                       lr: float, 
                       num_epochs: int,
                       gradient_fn) -> np.ndarray:
    
    N, num_features = X.shape
    losses = [loss(X, Y, theta0)]
    theta = theta0
    for _ in range(num_epochs):
        for i in range(N):
            X_i = X[i].reshape((1, num_features))
            Y_i = Y[i].reshape((1, 1))
            theta = theta - lr*gradient_fn(X_i, Y_i, theta)
        losses.append(loss(X, Y, theta))
    losses = np.asarray(losses)
    fig=plt.plot(np.arange(len(losses)), losses, label='SGD for lr=%s' % lr)
    plt.xlabel("Epochs")
    plt.ylabel("Loss", rotation=0, labelpad=25)
    plt.legend(loc='upper right')
    
    return fig

def GD_gradient(X: np.ndarray, Y: np.ndarray, theta: np.ndarray)  -> np.ndarray:
    
    N, num_features = X.shape
    Y_hat = X@theta # shape (N, 1)
    gradients = (-1/N)*np.sum((Y - Y_hat)*X, axis=0) # shape (M+1,)
    gradients = np.reshape(gradients, (num_features, 1)) # shape (M+1,1)
    
    return gradients

def GD_update(theta: np.ndarray, gradients: np.ndarray, lr: float) -> np.ndarray:
    return theta - lr*gradients

def GD_train(X_train: np.ndarray, 
          Y_train: np.ndarray, 
          theta0: np.ndarray, 
          num_epochs: int, 
          lr: float) -> np.ndarray:
    
    theta = theta0
    for i in range(num_epochs):
        grad = GD_gradient(X_train, Y_train, theta)
        theta = GD_update(theta, grad, lr)    
    return theta

def visualize_GD_loss(X: np.ndarray, 
                       Y: np.ndarray, 
                       theta0: np.ndarray, 
                       lr: float, 
                       num_epochs: int,
                       gradient_fn) -> np.ndarray:
    
    N, num_features = X.shape
    losses = [loss(X, Y, theta0)]
    theta = theta0
    for _ in range(num_epochs):
        grad = GD_gradient(X, Y, theta)
        theta = GD_update(theta, grad, lr)    
        losses.append(loss(X, Y, theta))
    losses = np.asarray(losses)
    fig=plt.plot(np.arange(len(losses)), losses, label='GD for lr=%s' % lr)
    plt.xlabel("Epochs")
    plt.ylabel("Loss", rotation=0, labelpad=25)
    plt.legend(loc='upper right')
    
    return fig
        