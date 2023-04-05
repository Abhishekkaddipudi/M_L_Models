from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
dataset=load_breast_cancer()
X=dataset['data']
Y=dataset['target']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)
from sklearn.preprocessing import StandardScaler
st=StandardScaler()
from LogisticRegression import LogisticRegression
X_train=st.fit_transform(X_train)
X_test=st.transform(X_test)
cls=LogisticRegression(0.01,10000)
cls.fit(X_train,y_train)
y_pred=cls.predict(X_test)
print(accuracy_score(y_test,y_pred))
