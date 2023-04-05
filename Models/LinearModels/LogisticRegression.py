import numpy as np

class LogisticRegression:
    def __init__(self,learning_rate=0.01,iteration=1000) -> None:
        self.lr=learning_rate
        self.itr=iteration
        self.weights=None
        self.bias=None
        self.samples=None
        self.features=None
        self.sigmoid=lambda x:(1/(1+(np.exp(-x))))
    
    def _init_params(self):
        self.weights=np.zeros(self.features)
        self.bias=0

    def _update_params(self,dw,db):
        self.weights-=self.lr*dw
        self.bias-=self.lr*db

    def _calculate_cost(self,y,y_pred):
        y_1=(-y*np.log(y_pred))
        y_0=((1-y)*np.log(1-y_pred))   

        cost=y_1-y_0

        return np.sum(cost)/self.samples
    
    def _get_gradients(self,X,y,y_pred):
        error=y_pred-y
        dw=(1/self.samples)*np.dot(X.T,error)
        db=(1/self.samples)*np.sum(error)
        return dw,db
    
    def _get_prediction(self,X):
        linear_pred=np.dot(X,self.weights)+self.bias
        return self.sigmoid(linear_pred)
    
    def fit(self,X,y):
        
        self.samples,self.features=X.shape
        self._init_params()
        for i in range(self.itr):
            y_pred=self._get_prediction(X)
            dw,db=self._get_gradients(X,y,y_pred)
            cost=self._calculate_cost(y,y_pred)
            self._update_params(dw,db)
            


    def predict(self,X):
        y_pred=self._get_prediction(X)
        return [0 if y<0.5 else 1 for y in y_pred]

        