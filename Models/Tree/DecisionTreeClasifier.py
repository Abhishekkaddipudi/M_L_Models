import numpy as np
from collections import Counter
class Node:
    def __init__(self,features=None,threshold=None,left=None,right=None,*,value=None):
        self.features=features
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value
    def is_leaf_node(self):
        return self.value is not None
class DecisionTreeClassifier:
    def __init__(self,n_features=None,min_sample_split=2,max_depth=100):
        self.root=None
        self.min_sample_split=min_sample_split
        self.max_depth=max_depth
        self.n_features=n_features
    
    
    def fit(self,X,y):
        self.n_features=X.shape[1] if not self.n_features else min(self.n_features,X.shape[1])
        self.root=self._Tree(X,y)



    def _Tree(self,X,y,depth=0):
        n_samples,n_feats=X.shape
        n_labels=len(np.unique(y))

        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_sample_split):
            leaf_value=self._high_prob_class(y)
            return Node(value=leaf_value)
        feat_idxs=np.random.choice(n_feats,self.n_features,replace=False)

        best_feat,best_threshold=self._best_split(X,y,feat_idxs)
        left_idxs,right_idxs=self._split(X[:,best_feat],best_threshold)
        left=self._Tree(X[left_idxs,:],y[left_idxs],depth+1)
        right=self._Tree(X[right_idxs,:],y[right_idxs],depth+1)
        return Node(best_feat,best_threshold,left,right)
        



    def _best_split(self,X,y,feat_idxs):
        best_gain=-1
        split_idx,split_threshold=None,None
        for feat_idx in feat_idxs:
            X_column=X[:,feat_idx]
            thresholds=np.unique(X_column)
            for threshold in thresholds:
                gain=self._information_gain(X_column,y,threshold)
                if gain>best_gain:
                    best_gain=gain
                    split_idx=feat_idx
                    split_threshold=threshold
        return split_idx,split_threshold
    def _information_gain(self,X_column,y,threshold):
        parent_entrophy=self._entrophy(y)
        left_idxs,right_idxs=self._split(X_column,threshold)
        if len(left_idxs)==0 or len(right_idxs)==0:
            return 0
        n=len(y)
        n_l,n_r=len(left_idxs),len(right_idxs)
        e_l,e_r=self._entrophy(y[left_idxs]),self._entrophy(y[right_idxs])
        
        child_entrophy=(n_l/n)*e_l + (n_r/n)*e_r
        return parent_entrophy-child_entrophy
    def _split(self,X_column,threshold):
        left_idxs=np.argwhere(X_column<=threshold).flatten()
        right_idxs=np.argwhere(X_column>threshold).flatten()
        return left_idxs,right_idxs
    
    def _entrophy(self,y):
        hist=np.bincount(y)
        ps=hist/len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])
        

    def _high_prob_class(self,y):
        count=Counter(y)
        return count.most_common(1)[0][0]

    def predict(self,X):
        return np.array([self._traverse(x,self.root) for x in X])
    
    def _traverse(self,x,node):
        if node.is_leaf_node():
            return node.value
        if x[node.features]<=node.threshold:
            return self._traverse(x,node.left)
        return self._traverse(x,node.right)
    

        
        