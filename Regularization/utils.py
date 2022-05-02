"""M4 functions that are utilized in M5"""
from typing import Dict, Tuple 

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

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


    return token2index

def load_data(filepath: str, token2index: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    num_features = len(token2index) 
    with open(filepath, 'r') as f:
        for line in f.readlines():
            label, review = line.strip().split('\t') 

            # create feature vector 
            X_i = np.zeros(num_features)
            for word in review.split(' '):
                if  word in token2index:
                    X_i[token2index[word]] = 1 
            X.append(X_i)
            y.append([int(label)]) 

    return (np.array(X), np.array(y))# Implement load_data/feature engineering

def plot_coeffs_and_CV(X_train, Y_train, cv, penalty):    
    alphas = np.logspace(-4, 5, num=15, base=np.e)
    
    #Need to count the number of examples to properly find lambda from alpha, plus amount of CV for leave one out
    N = len(X_train)

    # plot coefficent size vs log(alpha) graph. 
    lasso = LogisticRegression(solver='liblinear', random_state=RANDOM_SEED, max_iter=10000, penalty=penalty)
    coefs = []
    for a in alphas:
        lasso.set_params(C=a)
        lasso.fit(X_train, Y_train)
        coefs.append(lasso.coef_.ravel().copy())
    ax = plt.gca()
    ax.plot(np.log(alphas), coefs)
    ax.set_xscale('linear')
    plt.axis('tight')
    plt.xlabel('log(C)')
    plt.ylabel('Coefficients')
    plt.title('Coefficient plot for Movie Dataset')


    # LassoCV: coordinate descent
    # Compute paths to find proper alpha
    tuned_parameters = [{'C': alphas}]
    n_folds = cv
    clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
    clf.fit(X_train, Y_train)
    accuracy = clf.cv_results_['mean_test_score']
    accuracy = np.abs(accuracy)
    plt.figure().set_size_inches(8, 3)
    plt.plot(np.log(alphas), accuracy)
    plt.ylabel('Accuracy')
    plt.xlabel('log(C)')
    plt.title('CV plot for Accuracy')
    plt.xlim([np.log(alphas[0]), np.log(alphas[-1])])