from LinearRegression import LinearRegression
from Ridge import Ridge
from Lasso import Lasso
import numpy as np
class ElasticPenalty:
    
    def __init__(self, l = 0.1, l_ratio = 0.5):
        self.l = l 
        self.l_ratio = l_ratio

    def __call__(self, w):
        l1_contribution = self.l_ratio * self.l * np.sum(np.abs(w))
        l2_contribution = (1 - self.l_ratio) * self.l    * np.sum(np.square(w))
        return (l1_contribution + l2_contribution)

    def derivation(self, w):
        l1_derivation = self.l * self.l_ratio * np.sign(w)
        l2_derivation = self.l * (1 - self.l_ratio) * w
        return (l1_derivation + l2_derivation)

class ElasticNet(LinearRegression):
    def __init__(self,l,l_ratio,learning_rate=0.01,iteration=3000) -> None:
        self.regularization=ElasticPenalty(l,l_ratio)
        super().__init__(learning_rate,iteration)

    def _calculate_cost(self, y, y_pred):
        return super()._calculate_cost(y, y_pred)+self.regularization(self.weights)
    def fit(self,X,y):
        self.samples,self.features=X.shape
        super()._init_params()
        for i in range(self.itr):
            y_pred=super()._get_prediction(X)
            cost=self._calculate_cost(y,y_pred)
            dw,db=super()._get_gradient(X,y,y_pred)
            dw+=self.regularization.derivation(self.weights)
            super()._update_params(dw,db)
            if i % 100 ==0:
                print(f"The Cost in iteration {i}----->{cost}")


        