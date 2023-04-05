import numpy as np
from collections import Counter

class KNN_Classifier:
    def __init__(self,k=3) -> None:
        self.k=k
    def _eucldean_distance(self,X1,X2):
        return np.sqrt(np.sum((X1-X2)**2))
    def _predict(self,X1):
        distance=[self._eucldean_distance(X1,X2) for X2 in self.X_train]

        k_idxs=np.argsort(distance)[:self.k]
        k_values=self.y_train[k_idxs]

        count=Counter(k_values)
        return count.most_common()[0][0]
    def predict(self,X):
        prediction=[self._predict(x) for x in X]
        return prediction
    def fit(self,X,y):
        self.X_train=X
        self.y_train=y
        
        
