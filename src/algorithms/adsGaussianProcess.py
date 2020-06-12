import os
import math
import pickle
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from adsUtils import create_pickle
from adsGPUtils import plot_gp, kernel
from adsGPUtils import posterior_predictive, nll_fn

from adsConfig import config

if os.path.exists('data.pkl') == False:
    PICKLE_FILE = create_pickle("data.json")

filename = 'data.pkl'

class GPmodel():
    def __init__(self, datafile = filename):
        
        with open("data.pkl", "rb") as f:
            try:
                self.FullData = pickle.load(f)
            except EOFError:
                print("File not Found")
                pass
        
        configuration = config()
        configData = configuration.GPconfig

        self.data = np.array(self.FullData['value'])
        self.data_mean = np.mean(self.data)
        self.data = self.data - self.data_mean
        
        '''
        Please refer the adsConfig.py file for descriptions of following variables.
        '''
        self.fit_required = configData.fit_required
        self.noise = configData.noise
        self.confidence = configData.confidence
        self.l = configData.l
        self.sigma_f = configData.sigma_f
        self.data_grain_size = configData.data_grain_size
        self.epsilon = configData.epsilon
        self.training_done = False

        self.support = np.arange(0, math.floor(1.1*len(self.data)), self.data_grain_size).reshape(-1, 1)

        self.mu_prior = np.zeros(self.support.shape)
        self.cov_prior = kernel(self.support, self.support)

        self.model_type = "Predictive"


    def train(self):
        self.X_train = np.arange(0, len(self.data)).reshape(-1, 1)
        self.Y_train = self.data
        
        if self.fit_required == True:
            res = minimize(nll_fn(self.X_train, self.Y_train, self.noise), [1, 1], bounds=((1e-5, None), (1e-5, None)), method='L-BFGS-B')
            self.l_opt, self.sigma_f_opt = res.x
            self.mu_posterior, self.cov_posterior = posterior_predictive(self.support,
                                                                         self.X_train,
                                                                         self.Y_train,
                                                                         l = self.l_opt,
                                                                         sigma_f = self.sigma_f_opt, 
                                                                         sigma_y = self.noise, 
                                                                         epsilon = self.epsilon)
        else:
            self.mu_posterior, self.cov_posterior = posterior_predictive(self.support,
                                                                         self.X_train,
                                                                         self.Y_train,
                                                                         l = self.l,
                                                                         sigma_f = self.sigma_f, 
                                                                         sigma_y = self.noise, 
                                                                         epsilon = self.epsilon)

        self.uncertainity = 1.96 * np.sqrt(np.diag(self.cov_posterior)) 
        self.training_done = True 


    def SampleFromDistribution(self,  numSamples = 3, posterior = True):
        if posterior == True:
            try:
                samples = np.random.multivariate_normal(self.mu_posterior.ravel(), self.cov_posterior, numSamples)
            except:
                print("ERROR: Please train the GP model using .train() method before sampling from posterior\n")

            plot_gp(self.mu_posterior.ravel(),
                    self.cov_posterior,
                    self.support, 
                    X_train=self.X_train, 
                    Y_train=self.Y_train, 
                    samples=samples)
        
        else:
            samples = np.random.multivariate_normal(self.mu_prior.ravel(), self.cov_prior, numSamples)
            plot_gp(self.mu_prior, self.cov_prior, self.support, samples=samples)


    def classify(self, test_data_point):
        if self.training_done == True:
            upper_bound_list = self.mu_posterior + self.uncertainity
            lower_bound_list = self.mu_posterior - self.uncertainity
            prediction_index = list(self.support).index(len(self.data))

            self.upper_bound = upper_bound_list[prediction_index] + self.data_mean
            self.lower_bound = lower_bound_list[prediction_index] + self.data_mean
            
            return (test_data_point < self.lower_bound or test_data_point > self.upper_bound)

        else:
            print("ERROR: Please train the GP model using .train() method before classifying the data point\n")


    def PlotUncertainity(self):
        if self.training_done == True:
            plt.plot(self.support, self.mu_posterior + self.uncertainity, 'r')
            plt.plot(self.support, self.mu_posterior - self.uncertainity, 'g')
            plt.show()
        else:
            print("ERROR: Please train the GP model using .train() method before plotting posterior uncertainity\n")

#Following lines runs only when this script is run independently, hence only used for testing	
if __name__ == '__main__':
    GP_model = GPmodel(filename)
    GP_model.train()
    GP_model.SampleFromDistribution()
    GP_model.PlotUncertainity()
