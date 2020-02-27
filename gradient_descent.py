import  numpy as np
class LinearRegression :
    base_t = 0
    base_i = 0
    def _gradient_descent(self,theta,intercept,alpha,X,Y):
        n= len(X[0])
        m= len(Y)
        x_intercept = np.array([1 for j in range(m)])
        for i in range(m) :
            h = np.dot(X,theta)
            intercept = intercept - alpha*(np.dot(x_intercept,h-Y))
            new_x = np.transpose(X)
            theta = theta - (alpha*np.dot(new_x,h-Y))
        return theta,intercept

    def fit(self,X,Y):
        m= X.shape[0]
        if len(X.shape)>1 :
            n = X.shape[1]
        else :
            n=1
        theta = np.zeros((n, 1))
        alpha = 0.0001
        intercept = 0
        LinearRegression.base_t,LinearRegression.base_i = self._gradient_descent(theta,intercept,alpha,X,Y)

        return self


    def predict(self,X):
        n_samples= X.shape[0]

        if n_samples== 1 :

            return (np.dot(X.T,LinearRegression.base_t)+ LinearRegression.base_i)
        else :
            r = [0]*n_samples
            for i in range(n_samples) :
                r[i] = np.dot(X[i].T,LinearRegression.base_t) + LinearRegression.base_i

            return r





        

