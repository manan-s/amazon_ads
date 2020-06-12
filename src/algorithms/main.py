from adsGaussianProcess import GPmodel
from adsARIMA import ARIMAmodel
from adsES import ESmodel

import numpy as np 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test_value', type = float,
                    help = 'Input value to be classified')
parser.add_argument('--use_GP', action = 'store_true', default = True,
                    help = "Whether to use GP model")
parser.add_argument('--use_ARIMA', action = 'store_true', default = True,
                    help = "Whether to use ARIMA model")
parser.add_argument('--use_ES', action = 'store_true', default = True,
                    help = "Whether to use ES model")

args = parser.parse_args()

test_value = args.test_value
filename = 'data.pkl'

model_GP = GPmodel(filename)
model_AR = ARIMAmodel(filename)
model_ES = ESmodel(filename, model_type = 'double')

def train():
    
    model_GP.train()
    model_AR.train()
    model_ES.train()

def classify(test_value):
    '''
    Classifies whether the test_value is anomalous

    Args:
        test_value: Float, a point that is to be classified as anomalous or non-anomalous
    
    Returns:
        Boolean: 1 if anomalous, 0 if non-anomalous  
    '''
    GP_prediction = int(model_GP.classify(test_value))
    AR_prediction = int(model_AR.classify(test_value))
    ES_prediction = int(model_ES.classify(test_value))
    
    '''
    Following makes sure atleast 2 algorithms agree that the point is an anomaly. This is an
    attempt to reduce false positives.
    '''
    return ((GP_prediction + AR_prediction + ES_prediction) >= 2)

train()
print(classify(test_value))
