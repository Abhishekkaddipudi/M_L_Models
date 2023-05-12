from DecisionTreeClasifier import DecisionTreeClassifier
import numpy as np
from collections import Counter

class RandomForest:
    def  __init__(self,n_trees=10,max_depth=10,min_sample_split=2,n_features=None) -> None:
        self.n_features=n_features
        self.min_sample_split=min_sample_split
        self.max_depth=max_depth
        self.n_trees=n_trees
        self.Trees=[]
    def fit(self,X,y):
        n_sample=X.shape[1]
        for _ in range(self.n_trees):
            tree=DecisionTreeClassifier(self.n_features,self.min_sample_split,self.max_depth)
            idxs=np.random.choice(n_sample,n_sample,replace=True)
            X_sample,y_sample=X[idxs],y[idxs]
            tree.fit(X_sample,y_sample)
            self.Trees.append(tree)


    def most_common(self,y):
        return Counter(y).most_common(1)[0][0]
    

    def predict(self,X):
        predictions=np.array([tree.predict(X) for tree in self.Trees])
        tree_pred=np.swapaxes(predictions,0,1)
        return np.array([self.most_common(pred) for pred in tree_pred])