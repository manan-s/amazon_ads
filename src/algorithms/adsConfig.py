'''
Configuration file for all models
'''
class config():
    def __init__(self):
        self.GPconfig = self.GPconfig()
        self.ARconfig = self.ARIMAconfig()
        self.ESconfig = self.ESconfig()

    class GPconfig():
        def __init__(self):
            '''
            Variable description:
            fit_required:    This is to get the maximum likelihood estimate of the
                             parameters of the RBF kernel (l and sigma). If this is 
                             set, the provided value of above will be ignored.
            
            noise:           Assumed noise level in the observations.
           
            confidence:      The confidence interval we are interested in.
            
            l:               RBF kernel parameter, controls horizontal variation. Higher
                             value leads to smoother functions
            
            sigma_f:         RBF kernel parameter, controls vertical variation. Alters the
                             confidence intervals.
                     
            data_grain_size: The finest interval size the GP is trained on.
            
            epsilon:         The amount we add to the Kss matrix for numerical stability.
            '''
            
            self.fit_required = False
            self.noise = 0.8
            self.confidence = 0.95
            self.l = 4
            self.sigma_f = 2.0
            self.data_grain_size = 0.1
            self.epsilon = 1e-8
            
    class ARIMAconfig():
        def __init__(self):
            '''
            Variable description:
            param_p:            This is the size of lookback window for the number of observations. (AR)
            
            param_q:            This is the amount of differencing needed to be done before training.
            
            param_q:            This is the size of lookback window for the past errors seen. (MA)
            
            describe_residuals: If set, this will print out the residuals.
            
            train_data_split:   For training purposes. The amount of split in fractions.
            '''
            
            self.param_p = 7
            self.param_d = 1
            self.param_q = 2
            self.describe_residuals = False
            self.train_data_split = 0.8
    
    class ESconfig():
        def __init__(self):
            '''
            Variable description:
            param_alpha: Smoothing parameter, lies in [0,1]. A value of 0 results in forecast
                         to be the average of previous forecast, a value of 1 makes forecast
                         equals to last observation.
            
            param_beta:  Trend smoothing parameter, lies in [0,1].
            
            param_gamma: Seasonality smoothing parameter, must be an integer greater than 1.
            
            param_phi:   Damping parameter. Lies in (0,1). Reduces variance in forecast.
            
            std_dev:     Difference between forecast and any of the bounds. Needs to be tuned
                         for Amazon's dataset
            '''
            
            self.param_alpha = 0.8
            self.param_beta = 0.2
            self.param_gamma = 4
            self.param_phi = 0.5
            self.std_dev = 5.0
