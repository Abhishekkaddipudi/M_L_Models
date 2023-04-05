import numpy as np
import pickle as pkl

class LinearRegression:
    def __init__(self,learning_rate=0.1,iteration=5000):

        self.itr=iteration
        self.lr=learning_rate
        self.weights=None
        self.bias=None
        self.samples=0
        self.features=0
    def _init_params(self):
        self.weights=np.zeros(self.features)
        self.bias=0
    def _calculate_cost(self, y, y_pred):
        return (1 / (2*self.samples)) * np.sum(np.square(y_pred-y))
    def _get_prediction(self,X):
        return np.dot(X,self.weights)+self.bias
    def _update_params(self,dw,db):
        self.weights-=self.lr*dw
        self.bias-=self.lr*db
    def _get_gradient(self,X,y,y_pred):
        error=y_pred-y
      
        dw=(1/self.samples)*np.dot(X.T,error)
        db=(1/self.samples)*np.sum(error)
        return dw,db        
    def fit(self,X,y):
        self.samples,self.features=X.shape
        self._init_params()
        for i in range(self.itr):
            y_pred=self._get_prediction(X)
            cost=self._calculate_cost(y,y_pred)
            dw,db=self._get_gradient(X,y,y_pred)
            self._update_params(dw,db)
            if i % 100 == 0:
                print(f"The Cost in iteration {i}----->{cost}")
    def predict(self,X):
        return self._get_prediction(X)

 
    def save_model(self):
        file = open('Linear_regression.pkl', 'wb')
        pkl.dump(self,file)
        file.close()