import numpy as np
class SGD_Linear_Regression :
    def __init__(self,max_iter =100):
        self.max_iter  = max_iter

    def _SGD(self,x,y):
        dim = x.ndim
        x_b = np.c_[np.ones((len(x),1)),x]
        dim = x.ndim
        if dim ==1 :
            raise ValueError('expected 2d ndarray given 1d ndarray')
        else:
            m,n = x.shape
            self.theta = np.zeros((n+1,1))
            self.theta[0][0] =1
            alpha = 0.0001
            for iter in range(self.max_iter) :
                for i in range(m) :
                    self.theta = self.theta - alpha*(x_b[i].reshape(1,-1).dot(self.theta) - y[i])*(x_b[i].reshape(-1,1))


    def fit(self,x,y):
        self._SGD(x,y)
        return self

    def predict(self,x):
        dimension = x.ndim
        if dimension ==1 :
            raise ValueError('expected 2d array given 1d array')
        else :
            n_samples ,n_features = x.shape
            x_a = np.c_[np.ones((n_samples,1)),x]
            try:
                h = np.dot(x_a,self.theta)
                return h
            except :
                raise NotImplementedError('This LinearRegression is not fitted yet.')

