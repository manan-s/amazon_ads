#implement adfuller
import os
import sys
import math
import pickle
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from pandas import DataFrame
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

from adsUtils import create_pickle
from adsConfig import config

filename = 'data.pkl'

class ESmodel():
    def __init__(self, datafile = filename, model_type = "double"):

        if os.path.exists(filename) == False:
            PICKLE_FILE = create_pickle("data.json")

        with open(filename, "rb") as f:
            try:
                self.FullData = pickle.load(f)
            except EOFError:
                print("File not Found")
                pass
        
        configuration = config()
        configData = configuration.ESconfig

        self.data = np.array(self.FullData['value'])
        self.data_mean = np.mean(self.data)

        self.param_alpha = configData.param_alpha
        self.param_beta = configData.param_beta
        self.param_gamma = configData.param_gamma
        self.param_phi = configData.param_phi
        self.std_dev = configData.std_dev

        self.es_type = model_type
        self.training_done = False

    def train(self):

        if self.es_type == "simple":
            self.fit = SimpleExpSmoothing(self.data).fit()

            self.forecast = self.fit.forecast(1)[0]

        elif self.es_type == "double":
            self.fit1 = Holt(self.data).fit(smoothing_level = self.param_alpha,
                                       smoothing_slope = self.param_beta,
                                       optimized = False)

            self.fit2 = Holt(self.data, exponential = True).fit(smoothing_level = self.param_alpha,
                                                                smoothing_slope = self.param_beta,
                                                                optimized = False) 

            self.fit3 = Holt(self.data, damped = True).fit(smoothing_level = self.param_alpha,
                                                            smoothing_slope = self.param_beta,
                                                            optimized = True)

            forecast1 = self.fit1.forecast(1)[0]
            forecast2 = self.fit2.forecast(1)[0]
            forecast3 = self.fit3.forecast(1)[0]
            self.forecast = (forecast1 + forecast2 + forecast3)/3.0

        elif self.es_type == "triple":
            #WARNING: Fails to converge with stock price data

            self.fit1 = ExponentialSmoothing(self.data,
                                             seasonal_periods = self.param_gamma,
                                             trend='add',
                                             seasonal='add').fit(use_boxcox=True)

            self.fit2 = ExponentialSmoothing(self.data,
                                             seasonal_periods = self.param_gamma,
                                             trend='add',
                                             seasonal='mul').fit(use_boxcox=True)
            
            self.fit3 = ExponentialSmoothing(self.data,
                                             seasonal_periods = self.param_gamma,
                                             trend='add',
                                             seasonal='add',
                                             damped = True).fit(use_boxcox=True)
            
            self.fit4 = ExponentialSmoothing(self.data,
                                             seasonal_periods = self.param_gamma,
                                             trend='add',
                                             seasonal='mul',
                                             damped = True).fit(use_boxcox=True)
            
            forecast1 = self.fit1.forecast(1)[0]
            forecast2 = self.fit2.forecast(1)[0]
            forecast3 = self.fit3.forecast(1)[0]
            forecast4 = self.fit4.forecast(1)[0]
            self.forecast = (forecast1 + forecast2 + forecast3 + forecast4)/4.0

        else:
            print("ERROR: Invalid argument.")
            sys.exit()
        
        self.training_done = True

    def classify(self, test_data_point):
        if self.training_done == True:
             
            self.lower_bound = self.forecast - 1.96*self.std_dev
            self.upper_bound = self.forecast + 1.96*self.std_dev
            
            return (test_data_point < self.lower_bound or test_data_point > self.upper_bound)

        else:
            print("ERROR: Please train the ES model using .train() method before classifying the data point\n")

    def plotFittedValues(self):

        if self.training_done == True:
            if self.es_type == "simple":
                self.fittedvalues = self.fit.fittedvalues
                
                pyplot.plot(self.fittedvalues, color = 'green')
                pyplot.plot(self.data, color = 'black')

                pyplot.show()
            
            elif self.es_type == "double":
                self.fittedvalues1 = self.fit1.fittedvalues
                self.fittedvalues2 = self.fit2.fittedvalues
                self.fittedvalues3 = self.fit3.fittedvalues

                pyplot.plot(self.fittedvalues1, color = 'red')
                pyplot.plot(self.fittedvalues2, color = 'green')
                pyplot.plot(self.fittedvalues3, color = 'blue')
                pyplot.plot(self.data, color = 'black')

                pyplot.show()

            else:
                self.fittedvalues1 = self.fit1.fittedvalues
                self.fittedvalues2 = self.fit2.fittedvalues
                self.fittedvalues3 = self.fit3.fittedvalues
                self.fittedvalues4 = self.fit4.fittedvalues
                
                pyplot.plot(self.fittedvalues1, color = 'red')
                pyplot.plot(self.fittedvalues2, color = 'green')
                pyplot.plot(self.fittedvalues3, color = 'blue')
                pyplot.plot(self.fittedvalues4, color = 'orange')
                pyplot.plot(self.data, color = 'black')

                pyplot.show()
                
        else:
            print("ERROR: Please train the ES model using .train() method before plotting the fitted values\n")

if __name__ == '__main__':
    DS_model = ESmodel(datafile = filename, model_type = 'double')
    DS_model.train()
    print(DS_model.classify(54.4))
    print(DS_model.forecast)
    DS_model.plotFittedValues()