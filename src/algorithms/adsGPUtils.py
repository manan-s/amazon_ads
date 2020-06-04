import pickle
import numpy as np
from scipy.optimize import minimize
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv, cholesky, det, lstsq


def plot_gp(mu, cov, support, X_train=None, Y_train=None, samples=[]):
    '''
    plot_gp: Plots the posterior predictive of Data
    Args:
        mu: mean function
        cov: covariance matrix
        support: The values in some range in X-axis we're working on
        samples: The list of samples from distribution parametrized by mu, cov
    
    Returns:
        Plots the GP distribution over functions with mean and some samples
    '''
    support = support.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    
    plt.fill_between(support, mu + uncertainty, mu - uncertainty, alpha=0.1)

    plt.plot(support, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(support, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()
    plt.show()

def plot_gp_2D(gx, gy, mu, X_train, Y_train, title, i):
    ax = plt.gcf().add_subplot(1, 2, i, projection='3d')
    ax.plot_surface(gx, gy, mu.reshape(gx.shape), cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
    ax.scatter(X_train[:,0], X_train[:,1], Y_train, c=Y_train, cmap=cm.coolwarm)
    ax.set_title(title)

def kernel(X1, X2, l=2.0, sigma_f=5.0):
    '''
    Isotropic squared exponential kernel. Computes 
    a covariance matrix from points in X1 and X2.
        
    Args:
        X1: Array of m points (m x 1).
        X2: Array of n points (n x 1).

    Returns:
        Covariance matrix (m x n).
    '''
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

def posterior_predictive(X_s, X_train, Y_train, l=50.0, sigma_f=1.0, sigma_y=1e-8, epsilon=1e-8):
    '''  
    Computes the suffifient statistics of the GP posterior predictive distribution 
    from m training data X_train and Y_train and n new inputs X_s.
    
    Args:
        X_s: New input locations (n x 1).
        X_train: Training locations (m x 1).
        Y_train: Training targets (m x 1).
        l: Kernel length parameter.
        sigma_f: Kernel vertical variation parameter.
        sigma_y: Noise parameter.
    
    Returns:
        Posterior mean vector (n x 1) and covariance matrix (n x n).
    '''
    '''                       __                 __
                             | K    K_s(transpose) |
     FullCovarianceMatrix =  | K-s      K_ss       |
                             |__                 __|
    '''
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + epsilon * np.eye(len(X_s))
    K_inv = inv(K)
    
    mu_posterior = K_s.T.dot(K_inv).dot(Y_train)
    cov_posterior = K_ss - K_s.T.dot(K_inv).dot(K_s)
    
    return mu_posterior, cov_posterior

def nll_fn(X_train, Y_train, noise, naive=True):
    '''
    Returns a function that computes the negative log marginal
    likelihood for training data X_train and Y_train and given 
    noise level.
    
    Args:
        X_train: training locations (m x d).
        Y_train: training targets (m x 1).
        noise: known noise level of Y_train.
        naive: if True use a naive implementation of negative log likelihood
        function, if False use a numerically more stable implementation. 
        
    Returns:
        Minimization objective.
    '''
    
    def nll_naive(theta):
        '''
        Function that calculates negative log likelihood

        Args:
            theta: A tuple of data parameters, in this case, (XTrain, YTrain, noise)
        
        Returns:
            Value of negative log likelihood of function
        '''
        K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
            noise**2 * np.eye(len(X_train))
        return 0.5 * np.log(det(K)) + \
               0.5 * Y_train.T.dot(inv(K).dot(Y_train)) + \
               0.5 * len(X_train) * np.log(2*np.pi)

    def nll_stable(theta):
        '''
        Function that calculates a more numerically stable negative log likelihood

        Args:
            theta: A tuple of data parameters, in this case, (XTrain, YTrain, noise)
        
        Returns:
            Value of negative log likelihood of function
        '''
        K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
            noise**2 * np.eye(len(X_train))
        L = cholesky(K)
        return np.sum(np.log(np.diagonal(L))) + \
               0.5 * Y_train.T.dot(lstsq(L.T, lstsq(L, Y_train)[0])[0]) + \
               0.5 * len(X_train) * np.log(2*np.pi)
    
    if naive:
        return nll_naive
    else:
        return nll_stable