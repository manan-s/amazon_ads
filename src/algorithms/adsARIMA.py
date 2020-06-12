import os
import math
import pickle
import numpy as np

from pandas import DataFrame
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from adsUtils import create_pickle
from adsConfig import config

if os.path.exists('data.pkl') == False:
    PICKLE_FILE = create_pickle("data.json")

filename = 'data.pkl'

class ARIMAmodel():
    def __init__(self, filename = filename):
        
        with open("data.pkl", "rb") as f:
            try:
                self.FullData = pickle.load(f)
            except EOFError:
                print("File not Found")
                pass
        
        configuration = config()
        configData = configuration.ARconfig

        self.data = np.array(self.FullData['value'])
        self.data_mean = np.mean(self.data)
	
	'''
        Please refer the adsConfig.py file for descriptions of following variables.
        '''
        self.param_p = configData.param_p
        self.param_d = configData.param_d
        self.param_q = configData.param_q
        self.describe_residuals = configData.describe_residuals
        self.train_data_split = configData.train_data_split

        self.training_done = False

    def PlotAutocorrelation(self):
        autocorrelation_plot(self.data)
        pyplot.show()

    def train(self):
        self.model = ARIMA(self.data,
		              order = (self.param_p, self.param_d, self.param_q)  )
        self.fitted_model = self.model.fit(disp = False)
        self.training_done = True

    def classify(self, test_data_point):
        if self.training_done == True:
            
            self.output = self.fitted_model.forecast()
            self.forecasted_value = self.output[0][0]
            self.std_dev = self.output[1][0]
            self.upper_bound = self.output[2][0][1]
            self.lower_bound = self.output[2][0][0]
            
            return (test_data_point < self.lower_bound or test_data_point > self.upper_bound)

        else:
            print("ERROR: Please train the ARIMA model using .train() method before classifying the data point\n")

    def plotResiduals(self, describe = False):
        if self.training_done == True:
            residuals = DataFrame(self.fitted_model.resid)
            print(residuals.describe())
            residuals.plot()
            pyplot.show()  
            residuals.plot()
            pyplot.show(kind = 'kde')    

            if describe == True:
                print(residuals.describe())
        
        else:
            print("ERROR: Please train the ARIMA model using .train() method before plotting the residuals\n")

    def VisualizeOnTrainingData(self):
        '''
        WARNING: A very slow operation
        '''
        size = int(self.train_data_split * len(self.data))
        train, test = self.data[0:size], self.data[size:len(self.data)]
        history = [data_point for data_point in train]

        predictions = list()

        for test_point_ind in range(len(test)):
            model = ARIMA(history,
		              order = (self.param_p, self.param_d, self.param_q))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()

            predicted_value = output[0]
            predictions.append(predicted_value)

            observed_value = test[test_point_ind]
            history.append(observed_value)

            print('predicted=%f, expected=%f' % (predicted_value, observed_value))
        
        error = mean_squared_error(test, predictions)
        print('Test MSE: %.3f' % error)
        
        pyplot.plot(test, color='green')
        pyplot.plot(predictions, color='red')
        pyplot.show()

#Following lines runs only when this script is run independently, hence only used for testing	
if __name__ == '__main__':
	test_point = 123.3
	
	AR_model = ARIMAmodel(filename)
	AR_model.train()
	AR_model.classify(test_point)
	AR_model.VisualizeOnTrainingData()
