import warnings 
warnings.filterwarnings("ignore")


from sklearn.datasets import load_breast_cancer
dataset=load_breast_cancer()
X=dataset['data']
Y=dataset['target']   
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.1,random_state=1234)
# from sklearn.preprocessing import StandardScaler
# st=StandardScaler()
# X_train=st.fit_transform(X_train)
# X_test=st.transform(X_test)
from RandomForest import RandomForest

for i in range(10,15):
    DR=RandomForest(max_depth=i)
    DR.fit(X_train,y_train)
    y_pred=DR.predict(X_test)
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test,y_pred))
