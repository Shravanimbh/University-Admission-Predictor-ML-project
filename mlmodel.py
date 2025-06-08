import pandas as pd
import numpy as np
import seaborn as sns

data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Admission_Prediction.csv')
data

data = data.drop("Serial No." , axis = 1)

data

data.isnull().sum()

data.isnull().sum().sum()

"""FILLING THE NULL VALUES"""

data = data.dropna(how = "any")

data.isnull().sum()

#split

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

x

y

from sklearn.model_selection import train_test_split

X_train, X_test , Y_train , Y_test = train_test_split(x,y,train_size = 0.70 , random_state = 7)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train , Y_train)

Y_pred_test = model.predict(X_test)
Y_pred_test

y_pred_train = model.predict(X_train)
y_pred_train

dfnew = pd.DataFrame(Y_train)
dfnew

dfpred = pd.DataFrame(y_pred_train)
dfpred

df3 = pd.concat([dfnew , dfpred] , axis = 1 , ignore_index = True )
df3.columns = ["Actual" , "Predicted"]
df3

from sklearn import metrics

print("R2Stat" , metrics.r2_score(Y_test , Y_pred_test))

def adj_r2(x,y):
    r2=model.score(x,y)
    print(r2)
    n=x.shape[0]
    p=x.shape[1]
    adjusted_r2=1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2

print("Adj r2 for training data",adj_r2(X_train,Y_train))
print("Adj r2 for testing data",adj_r2(X_test,Y_test))

print("mean_squared_error",metrics.mean_squared_error(Y_test,Y_pred_test))
print("mean_absolute_error",metrics.mean_absolute_error(Y_test,Y_pred_test))

print("Accuracy of the model",100-(np.mean(np.abs((Y_test-Y_pred_test)/Y_pred_test))*100))

model.coef_

model.intercept_


model.dump(model,"model.pkl" , 'wb')