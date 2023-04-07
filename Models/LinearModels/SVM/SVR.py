import numpy as np

class SVR:
    def __init__(self,lambda_param=0.01,learning_rate=0.01,iteration=3000):
        self.lr=learning_rate
        self.itr=iteration
        self.lambda_=lambda_param
    def _init_params(self):
        self.weights=np.zeros(self.features)
        self.bias=0
    def _get_params(self,X,y):
        for idx,x_i in enumerate(X):
            condition=y[idx]*(np.dot(x_i,   self.weights)-self.bias)>=1
            if condition:
                self.weights-=self.lr *(2 *self.lambda_*self.weights)
            else:
                self.weights-=self.lr *(2 *self.lambda_*self.weights-np.dot(x_i,y[idx]))
                self.bias-=self.lr*y[idx]

    def fit(self,X,y):
        self.samples,self.features=X.shape
        self._init_params()
        for i in range(self.itr):
            self._get_params(X,y)

    def predict(self,X):
        return np.dot(X,self.weights)-self.bias