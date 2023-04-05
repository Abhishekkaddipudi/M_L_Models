import warnings 
warnings.filterwarnings("ignore")


from sklearn.datasets import load_boston
dataset=load_boston()
X=dataset['data']
Y=dataset['target']   
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.1,random_state=1234)
from sklearn.preprocessing import StandardScaler
st=StandardScaler()
X_train=st.fit_transform(X_train)
X_test=st.transform(X_test)
from DecisionTreeRegressor import DecisionTreeRegression

for i in range(1,6):
    DR=DecisionTreeRegression(max_depth=i)
    DR.fit(X_train,y_train)
    y_pred=DR.predict(X_test)
    from sklearn.metrics import r2_score
   
    print(r2_score(y_test,y_pred))
