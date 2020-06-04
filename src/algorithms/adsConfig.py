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
            
            self.fit_required = False
            self.noise = 0.8
            self.confidence = 0.95
            self.l = 4
            self.sigma_f = 2.0
            self.data_grain_size = 0.1
            self.epsilon = 1e-8
            
    class ARIMAconfig():
        def __init__(self):
            
            self.param_p = 7
            self.param_d = 1
            self.param_q = 2
            self.describe_residuals = False
            self.train_data_split = 0.8
    
    class ESconfig():
        def __init__(self):

            self.param_alpha = 0.8
            self.param_beta = 0.2
            self.param_gamma = 4
            self.param_phi = 0.5
            self.std_dev = 5.0