"""Utilities functions"""
from typing import List, Tuple, Any, Dict

import os

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation, rc

def error_rate(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Given the true labels y and predicted label y_hat, returns the error rate
    
    Parameters
    ----------
    y: type `np.ndarray`, shape (N, 1)
        2D numpy array of true labels
    y_hat: type `np.ndarray`, shape (N, 1)
        2D numpy array of predicted labels
    """
    N = y.shape[0]
    num_incorrect = 0
    for (y_i, y_hat_i) in zip(y, y_hat):
        if y_i != y_hat_i: num_incorrect += 1
    return num_incorrect / N

def plot_data(X_train, y_train, dataset_name='toy', 
              fig_size=(6.4, 4.8), xlim=(-1, 8), ylim=(-1,4)):
    
    if not os.path.isdir('./figures'):
        os.mkdir('./figures')
    
    plt.figure(figsize=fig_size) # create new figure
    colors = ['#25B3F5', '#E98000'] # 'blue' and 'orange'

    # plot points  
    for y in np.unique(y_train):
        ix = np.where(y_train == y)
        plt.scatter(X_train[ix, 1], y_train[ix], color=colors[y], label='y=%d' % y)

    plt.grid()
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.yticks([i for i in range(ylim[0], ylim[1]+1)])
    plt.title('%s Dataset Visualization' % dataset_name)
    plt.xlabel('x')
    plt.ylabel('y', rotation=0)
    plt.legend(loc='lower right') # show legend

    plt.savefig('./figures/%s_dataset.png' % dataset_name) # save figure for comparison

def plot_nlls(num_epochs: List[int], train_nlls: List[float], val_nlls: List[float], fig_size=(6.4, 4.8)) -> None:
    
    if not os.path.isdir('./figures'):
        os.mkdir('./figures')

    # create a new figure, to avoid duplicating with figure in previous plots
    plt.figure(figsize=fig_size) 

    # add title and labels and title
    plt.title('Average Negative Log Likelihood vs. Number of Epochs')
    plt.xlabel('Number of epochs')
    # plt.xticks(num_epochs)
    plt.ylabel('Average negative log likelihood')
    
    # add train and error plots here: students write the below 2 lines
    plt.plot(num_epochs, train_nlls, label='training')
    plt.plot(num_epochs, val_nlls, label='validation')
    
    # show legends and save figure
    plt.legend() # show legend
    plt.savefig('./figures/nlls.png') # save figure for comparison

def plot_objective_contours(X, y, loss_fn, w_min=-6, w_max=6, title=None, colors=None,
        show_labels=True, new_figure=True, show_figure=True, save_filename=None, fig_size=(6.4, 4.8)):
    """
    Plots the logistic_regression.objective function with parameters
    X, y, lamb (lambda).

    X: Nx2 numpy ndarray, training input
    y: Nx1 numpy ndarray, training output
    lamb: Scalar lambda hyperparameter
    w_min (default=-8): Minimum of axes range
    w_max (default=8): Maximum of axes range
    title (default=None): Title of plot if not None
    colors (default=None): Color of contour lines. None will use default cmap.
    show_labels (default=True): Show numerical labels on contour lines
    new_figure (default=True): If true, calls plt.figure(), which create a 
        figure. If false, it will modify an existing figure (if one exists).
    show_figure (default=True): If true, calls plt.show(), which will open
        a new window and block program execution until that window is closed
    save_filename (defalut=None): If not None, save figure to save_filename 
    TODO: Change w's to theta's, as theta = [b, w1, ..., w_m]
    """
    N = 101
    
    w1 = np.linspace(w_min, w_max, N)
    w2 = np.linspace(w_min, w_max, N)
    W1, W2 = np.meshgrid(w1,w2)
    
    obj = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            w = np.array([[W1[i,j]], [W2[i,j]]])
            obj[i, j] = loss_fn(X, y, w)
    
    # Ploting contour
    if new_figure:
        plt.figure(figsize=fig_size)

    ax = plt.gca()
    contour_plot = ax.contour(W1, W2, obj, levels=12, colors=colors)
    if show_labels:
        ax.clabel(contour_plot, inline=1)
    plt.tick_params()
    ax.set_xlabel('theta1')
    ax.set_ylabel('theta2', rotation=0, labelpad=12)

    ax.axhline(0, color='lightgray')
    plt.axvline(0, color='lightgray')
    ax.set_axisbelow(True)

    if title is not None:
        plt.title(title)

    if save_filename is not None:
        plt.savefig(save_filename)

    if show_figure:
        plt.show()

def plot_optimization_path(point_list, color, linestyle='-', label=None):
    """
    Plot arrows stepping between points in the point list.

    point_list: List of 2D points, each of which is a 2x1 numpy ndarray
    color: matplotlib color
    linestyle: matplotlib linestyle
    label: Label to put in the plt.legend (plt.legend is not called in here)

    Does not call plt.figure() or plt.show()
    """
    X, Y, U, V = [], [], [], []

    start = point_list[0]
    for point in point_list[1:]:
        X.append(start[0])
        Y.append(start[1])

        U.append(point[0]-start[0])
        V.append(point[1]-start[1])

        start = point

    plt.quiver(X, Y, U, V, angles="xy", scale_units="xy", scale=1,
        color=color, linestyle=linestyle, linewidth=2, label=label)

def plot_sgd(X_train, y_train, compute_objective, theta_history, fig_size=(6.4, 4.8)):
    
    if not os.path.isdir('./figures'):
        os.mkdir('./figures')
    
    plot_objective_contours(X_train, y_train, compute_objective, w_min=-6, w_max=6, 
        new_figure=True, fig_size=fig_size, show_figure=False, save_filename=None)
    plot_optimization_path(theta_history, color='red', label='SGD')
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.title('Optimization Path of SGD on Contour Plot')

    plt.savefig('./figures/sgd.png')

def plot_decision_boundary(X_train, y_train, theta, fig_size=(6.4, 4.8), xlim=(-1, 8), ylim=(-1,4)): 

    if not os.path.isdir('./figures'):
        os.mkdir('./figures')

    # create new figure
    plt.figure(figsize=fig_size)

    colors = ['#25B3F5', '#E98000'] # 'blue' and 'orange'

    # plot points  
    y_train = y_train.flatten() # convert shape (N, 1) to shape (N,)
    for y in np.unique(y_train):
        ix = np.where(y_train == y)
        plt.scatter(X_train[ix, 1], y_train[ix], color=colors[y], label='y=%d' % y)

    # plot decision boundary line 
    x = np.linspace(-1,8,40)
    x_db = [- theta[0]/theta[1] for _ in range(x.shape[0])]
    plt.plot(x_db, x, '-r', color='#B92929', label='decision boundary')
    sigmoid = 1 / (1 + np.exp(-(theta[0] + theta[1]*x)))
    plt.plot(x, sigmoid, color='#414141', label='logistic curve')
    plt.grid()
    plt.legend(loc='lower right')
    plt.title('Logistic Regression Decision Boundary and Logistic (Sigmoid) Curve')
    plt.xlabel('x')
    plt.ylabel('y', rotation=0)
    plt.yticks([i for i in range(ylim[0], ylim[1]+1)])
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])

    plt.savefig('./figures/decision_boundary.png')

def make_animations(X_train, y_train, loss_fn, theta_history): 
    """
    Makes decision boundary and optimization path

    References:
    http://louistiao.me/posts/notebooks/embedding-matplotlib-animations-in-jupyter-as-interactive-javascript-widgets/    
    https://towardsdatascience.com/animations-with-matplotlib-d96375c5442c
    """
    # create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))

    # create contour plot
    N = 101
    w_min, w_max = -6, 6
    w1 = np.linspace(w_min, w_max, N)
    w2 = np.linspace(w_min, w_max, N)
    W1, W2 = np.meshgrid(w1,w2)
    
    obj = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            w = np.array([[W1[i,j]], [W2[i,j]]])
            obj[i, j] = loss_fn(X_train, y_train, w)

    contour_plot = ax1.contour(W1, W2, obj, levels=12, colors=None)
    ax1.clabel(contour_plot, inline=1) # show label
    ax1.tick_params()
    ax1.set_xlabel('theta1')
    ax1.set_ylabel('theta2', rotation=0, labelpad=12)
    ax1.axhline(0, color='lightgray')
    ax1.axvline(0, color='lightgray')
    ax1.set_axisbelow(True)
    ax1.set_title('Optimization Path of SGD on Contour Plot') 
    
    # set up points for quiver
    quiver = ax1.quiver([], [], [], [], angles='xy', scale_units='xy', scale=1,
                            color='red', linestyle='-', linewidth=2, label='SGD')
    
    # create decision boundary plot 
    ax2.grid()
    colors = ['#25B3F5', '#E98000'] # 'blue' and 'orange'
    y_train = y_train.flatten()
    for y in np.unique(y_train):
        ix = np.where(y_train == y)
        plt.scatter(X_train[ix, 1], y_train[ix], color=colors[y], label='y=%d' % y)
    
    # set up changing lines
    decision_boundary, = ax2.plot([], [], lw=2, color='#B92929', label='decision boundary')
    sigmoid, = ax2.plot([], [], lw=2, color='#414141', label='logistic curve')
    ax2.legend(loc='lower right')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y', rotation=0)
    xlim = (0, 6)
    ylim = (-1, 2)
    ax2.set_yticks([i for i in range(ylim[0], ylim[1]+1)])
    ax2.set_xlim(xlim[0], xlim[1])
    ax2.set_ylim(ylim[0], ylim[1])
    ax2.set_title('Logistic Regression Decision Boundary and Logistic (Sigmoid) Curve')

    def init():

        # set up (empty) decision boundary line
        decision_boundary.set_data([], [])
        sigmoid.set_data([], [])

        return (decision_boundary, sigmoid, quiver)

    # # animation function. This is called sequentially
    def animate(i):

        # plot optimization step
        X, Y, U, V = [], [], [], []

        start = theta_history[0]
        for point in theta_history[1:i+1]:
            X.append(start[0])
            Y.append(start[1])

            U.append(point[0]-start[0])
            V.append(point[1]-start[1])

            start = point
        quiver = ax1.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1,
                            color='red', linestyle='-', linewidth=2, label='SGD')

        # plot decision boundary line 
        x1 = np.linspace(-1,8,40)
        x2 = [- theta_history[i][0]/theta_history[i][1] for _ in range(x1.shape[0])]
        sigmoid_curve = 1 / (1 + np.exp(-(theta_history[i][0] + theta_history[i][1]*x1)))

        decision_boundary.set_data(x2, x1)
        sigmoid.set_data(x1, sigmoid_curve)

        return (decision_boundary, sigmoid, quiver)

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(theta_history), interval=60, blit=True)
    return anim