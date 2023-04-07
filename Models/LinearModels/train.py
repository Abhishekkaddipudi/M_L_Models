from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
dataset=load_iris()
X=dataset['data']
Y=dataset['target']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)

from MclassLogisticRegression import MclassLogisticRegression
cls=MclassLogisticRegression(0.01,4000)
cls.fit(X_train,y_train)
y_pred=cls.predict(X_test)
print(accuracy_score(y_test,y_pred))
