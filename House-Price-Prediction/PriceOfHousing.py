"""
Created on Wed Jul 25 12:49:26 2018
@author: Kina

Project 1 : 
    Predict the price of homes at sale for the Aimes Iowa Housing dataset
Goal :
    Predict the sales price for each house. For each Id in the test set, 
    you must predict the value of the SalePrice variable. 
Output :
    Id,SalePrice
    XX,XXXXX
    etc.  
"""

#Import the libraries
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from sklearn.metrics import r2_score

#Import dataset
#Import dataset
dataset = pd.read_csv('kc_house_data.csv')

X = dataset.iloc[:, 3:].values
y = dataset[['id','price']]
#y = dataset.iloc[:, 2].values

# Backward elimination (drop columns which have p-value > 0.05)
# get row count and column count
rows,columns = X.shape
X = np.append(arr=np.ones((rows, 1)).astype(int), values=X, axis=1)

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y['price'], x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    #print(regressor_OLS.summary())
    regressor_OLS.summary()
    return x
SL = 0.05
X_Modeled = backwardElimination(X, SL)

'''
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
print(regressor_OLS.summary())
'''

#Splitting the dateset into training set and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_Modeled,y,test_size = 0.3,random_state = 0)
# id is for output
y_test_id = y_test['id']
# price for prediction
y_train = y_train['price']
y_test = y_test['price']

#check accuracies
def checkAcc(regressor,var):
    # applying k-fold cross validation
    from sklearn.model_selection import cross_val_score
    accuracies =  cross_val_score(estimator = regressor,X = X_train,y = y_train,cv = 10)
    print('accuracies mean value of ' + var, accuracies.mean())
    print('accuracies std value of ' + var, accuracies.std())
    print('\n')

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor_PF = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor_PF.fit(X_train, y_train)
y_pred_RF = regressor_PF.predict(X_test)
print('R^2 of RandomForest: ' , r2_score(y_test,y_pred_RF))
checkAcc(regressor_PF,'RandomForest')

#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor_ML = LinearRegression()
regressor_ML.fit(X_train,y_train)
y_pred_ML = regressor_ML.predict(X_test)
print('R^2 of of MultipleLinearRegression: ' , r2_score(y_test,y_pred_ML))
checkAcc(regressor_ML,'MultipleLinearRegression')

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor_DT = DecisionTreeRegressor(random_state = 0)
regressor_DT.fit(X_train, y_train)
y_pred_DT = regressor_DT.predict(X_test)
print('R^2 of of DecisionTree: ' , r2_score(y_test,y_pred_DT))
checkAcc(regressor_DT,'DecisionTree')

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
regressor_POLY = PolynomialFeatures(degree = 2)
X_poly = regressor_POLY.fit_transform(X_train)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y_train)
y_pred_POLY = lin_reg.predict(regressor_POLY.fit_transform(X_test))
print('R^2 of of Polynomial: ' , r2_score(y_test,y_pred_POLY))
checkAcc(lin_reg,'Polynomial')

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = y_train.reshape(-1,1)
y_train = sc_y.fit_transform(y_train)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor_SVR = SVR(kernel = 'rbf')
regressor_SVR.fit(X_train, y_train)
#Predict
X_test = sc_X.fit_transform(X_test)
y_pred_SVR = regressor_SVR.predict(X_test)
y_pred_SVR = sc_y.inverse_transform(y_pred_SVR)
print('R^2 of SVR: ' , r2_score(y_test,y_pred_SVR))
checkAcc(regressor_SVR,'SVR')

#Output CSV files,using RandomForestRegressor's prediction result
output_csv = pd.DataFrame({'Id':y_test_id, 'SalePrice':y_pred_RF})
output_csv.to_csv('Prediction.csv', index=False)




