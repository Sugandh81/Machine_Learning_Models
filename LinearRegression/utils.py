import copy

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import ticker, cm
from sklearn.preprocessing import StandardScaler

def visualize_loss(X: np.ndarray, 
                   Y: np.ndarray, 
                   loss_fn,
                   full_annotate: bool = True,
                   save_fig: bool = False):
    cmap = cm.get_cmap('viridis') #colormap for plotting each step
    rgba = cmap(np.linspace(0,1,10))
    best_fit_color = rgba[0]
    
    theta = np.array([[-5],[-5]], ndmin = 2) #set the range of parameter values for this data
    theta0 = np.linspace(-theta[0] * 5, theta[0] * 5, 100) 
    theta1 = np.linspace(-theta[1] * 5, theta[1] * 5, 100)

    error = np.ones((100,100))
    for ix,i in enumerate(theta0): #for the grid of parameter values, calculate the true error and the student's error
        for jx,j in enumerate(theta1):
            error[ix, jx] = loss_fn(X, Y, np.array([i,j]))
        
    best_theta = np.unravel_index(np.argmin(error, axis=None), error.shape) 
    best_fit = np.array([theta0[best_theta[0]], theta1[best_theta[1]]]).reshape(-1,1) #find the best parameter fit among values calculated
    sort_paired_X = X[np.argsort(X[:, 0])] #Sort the dataset for nice line plot
    
    fig, ax = plt.subplots(1,2, figsize = (20,6)) #1x2 grid of plots
    #Create contour plot of the true error
    cs = ax[0].contour(theta0.flatten(), theta1.flatten(), error, linewidths = 1,cmap=cm.PuBu_r)
    ax[0].set_title('Contour Plot of Objective Function') 
    ax[0].clabel(cs, inline=1, fontsize=10) #label contour levels
    ax[0].set_xlabel('Slope Weight')
    ax[0].set_ylabel('Bias Term Weight')
    ax[0].plot(theta0[best_theta[0]],theta1[best_theta[1]],'k.') #plot dot at minimum error
    ax[0].set_xlim([-25,25])
    ax[0].set_ylim([-25,25])
      
    #Solution Line
    ax[1].set_title('Y vs X')
    ax[1].scatter(X[:,0],Y)
    ax[1].plot(sort_paired_X[:,0],sort_paired_X.dot(best_fit),color = best_fit_color, label = "Target Fit")
    ax[1].set_xlabel('X values')
    ax[1].set_ylabel('Y values')
    ax[1].set_xlim([-3,3])
    ax[1].set_ylim([-25,15])
    
    if full_annotate: #full_annotate defaults to true, but set to false for other visualizations to keep plots cleaner
        ax[0].annotate('Point @' + '\n' + 'minimum of objective func.' + '\n' + ' = (%.2f,%.2f)' % (best_fit[0][0],best_fit[1][0]),
                         (best_fit[0][0],best_fit[1][0]), xytext=(best_fit[0][0] + 0.5,best_fit[1][0] + 0.5))
        ax[1].text(-1, -7, 'Slope term @ ' + '\n' + 'minimum of objective func.' + '\n' + ' = %.2f' % best_fit[0][0], 
                  fontsize=10, rotation=-30, rotation_mode='anchor')
        ax[1].annotate('Bias term @ ' + '\n' + 'minimum of objective func.' + '\n' + ' = %.2f' % best_fit[1][0], 
                     (0, best_fit[1]), xytext=(0.5, best_fit[1] + 0.5),arrowprops=dict(facecolor='black', shrink=0.005))
    
    if save_fig: 
        fig.savefig('img/loss_and_fit.png')
    
    return (ax, fig)

def visualize_one_step(X: np.ndarray, 
                       Y: np.ndarray, 
                       theta0: np.ndarray, 
                       lr: float, 
                       loss_fn,
                       gradient_fn,
                       save_fig: bool = False) -> None:
    cmap = cm.get_cmap('viridis')
    rgba = cmap(np.linspace(0,1,10))
    
    init_color = rgba[9]
    one_step_color = rgba[7]
    ax, fig = visualize_loss(X, Y, loss_fn=loss_fn, full_annotate=False)
    theta_grad = gradient_fn(X, Y, theta0)
    
    #Zoom into arrows
    ax[0].set_xlim([-6,theta0[0] + 1])
    ax[0].plot(theta0[0], theta0[1], '.',color = init_color)  
    ax[0].annotate("initial weights", 
    		   xy=theta_grad*-1*lr+theta0, 
		   xytext=theta0, 
		   arrowprops=dict(arrowstyle="->", color = one_step_color))
    ax[0].annotate("weights after one epoch", 
    		     xy=theta_grad*-1*lr+theta0, 
		     xytext=theta_grad*-1*lr+theta0-1.2)
   
    #Sort the dataset for nice line plot
    sort_paired_X = X[np.argsort(X[:, 0])]
    
    #Solution Line
    ax[1].plot(sort_paired_X[:,0],
    	       sort_paired_X.dot(theta_grad*-1*lr+theta0),
	           color = one_step_color, 
               label = "Fit after one epoch")
    ax[1].plot(sort_paired_X[:,0],
    		  sort_paired_X.dot(theta0),
		      color = init_color, 
		      label = "Initial Fit")
    ax[1].legend()

    if save_fig: 
        fig.savefig('img/one_step_update.png')
    
def visualize_GD(X: np.ndarray, 
                 Y: np.ndarray, 
                 theta0: np.ndarray, 
                 lr: float, 
                 epochs: int,
                 loss_fn, 
                 gradient_fn,
                 update_fn, 
                 save_fig: bool = False) -> None:

    ax, fig = visualize_loss(X, Y, loss_fn, full_annotate=False)
    
    #Initialize two theta vectors since they'll be overwritten differently
    cmap = cm.get_cmap('viridis')
    rgba = cmap(np.linspace(0, 1, epochs))
    ax[0].set_xlim([-15, theta0[0] + 1])
    
    for i in range(epochs): #gradient descent for specified num of epochs
        color_idx = epochs - i - 1
        theta_grad = gradient_fn(X, Y, theta0)
        theta_update = update_fn(theta0, theta_grad, lr)
        ax[0].annotate("", xy=theta_update, xytext=theta0, arrowprops=dict(arrowstyle="->", color = rgba[color_idx]))
        ax[1].plot(X[:,0], X@theta_update, color = rgba[color_idx])
        theta0 = copy.deepcopy(theta_update)

    if save_fig: 
        fig.savefig('img/GD.png')

def visualize_predict(X: np.ndarray, 
                      Y: np.ndarray,
                      theta: np.ndarray, 
                      predict_fn, 
                      save_fig: bool = False) -> None: 
    
    Y_hat = predict_fn(X, theta)
    
    plt.plot(Y_hat, Y, '.') # Plot true vs predicted
    plt.plot(np.sort(Y), np.sort(Y)) # plot unity line
    
    plt.xlim([np.min(Y_hat)-1, np.max(Y_hat)+1]) # vary graph limits 
    plt.ylim([np.min(Y)-1, np.max(Y)+1])   
    
    plt.xlabel("Predicted Sound Pressure (dB)") # Label axes
    plt.ylabel("True Sound Pressure (dB)")

    if save_fig: 
        plt.savefig('img/predictions.png')