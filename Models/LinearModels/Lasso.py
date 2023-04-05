from LinearRegression import LinearRegression
import numpy as np

class Lassopenalty:
    def __init__(self,l):
        self.l=l
    def __call__(self,w):
        return self.l*np.sum(np.abs(w))
    def derivation(self,w):
        return self.l*np.sign(w)

class Lasso(LinearRegression):
    def __init__(self,l,learning_rate=0.1,iteration=5000):

        self.regularization=Lassopenalty(l)
        super().__init__(learning_rate,iteration)


    def _calculate_cost(self, y, y_pred):
        return super()._calculate_cost(y,y_pred)+self.regularization(self.weights)
    
    def fit(self,X,y):
        self.samples,self.features=X.shape
        super()._init_params()
        print(self.weights,self.bias)
        for i in range(self.itr):
            y_pred=super()._get_prediction(X)
            
            cost=self._calculate_cost(y,y_pred)
            dw,db=super()._get_gradient(X,y,y_pred)
            dw+=self.regularization.derivation(self.weights)
            self._update_params(dw,db)
            if i % 100 ==0:
                print(f"The Cost in iteration {i}----->{cost}")
    def predict(self,X):
        return super().predict(X)

