# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 14:10:40 2018
@author: Kina

Project 2 :
    Predict how capable each applicant is of repaying a loan
    
Goal:
    Predict clients' repayment abilities

OutPut:
    CSV file with SK_ID_CURR and TARGET

"""

#Import the libraries
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

#Import dataset
dataset_test = pd.read_csv(r'C:\Users\Kina\AI\Assignment2/application_test.csv')
dataset_train = pd.read_csv(r'C:\Users\Kina\AI\Assignment2/application_train.csv')

target_y = dataset_train['TARGET']
test_id = dataset_test['SK_ID_CURR']

#Delete SK_ID_CURR and TARGET
dataset_train.drop('SK_ID_CURR',axis = 1, inplace = True)
dataset_train.drop('TARGET',axis = 1, inplace = True)
dataset_test.drop('SK_ID_CURR',axis = 1, inplace = True)

# check missing values
# get categorical data
df_category_only_train = dataset_train.select_dtypes(exclude=[np.number])
df_category_only_test = dataset_test.select_dtypes(exclude=[np.number])

# check how many missing values exist
#df_category_only_train.isnull().sum()
#df_category_only_test.isnull().sum()
#drop all categorical columns with nan value
dataset_train = dataset_train.drop(['NAME_TYPE_SUITE',
                                          'OCCUPATION_TYPE',
                                          'FONDKAPREMONT_MODE',
                                          'HOUSETYPE_MODE',
                                          'WALLSMATERIAL_MODE',
                                          'EMERGENCYSTATE_MODE'],
                                          axis = 1)

dataset_test = dataset_test.drop(['NAME_TYPE_SUITE',
                                          'OCCUPATION_TYPE',
                                          'FONDKAPREMONT_MODE',
                                          'HOUSETYPE_MODE',
                                          'WALLSMATERIAL_MODE',
                                          'EMERGENCYSTATE_MODE'],
                                          axis = 1)


# convert 0 with a NaN
dataset_train.replace(0, np.nan, inplace = True)
dataset_test.replace(0, np.nan, inplace = True)


#drop entire row, minimum number of non-null values for the row:60
dataset_train = dataset_train.dropna(axis='rows', thresh = 60,how = 'all')
dataset_test = dataset_test.dropna(axis='rows', thresh = 60,how = 'all')

#dataset_train.isnull().sum()


#filling missing values with mean values
dataset_train = dataset_train.fillna(dataset_train.mean())
dataset_test = dataset_test.fillna(dataset_test.mean())

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
onehotencoder = OneHotEncoder()

for column in range(0-114):
    if dataset_train[column].type=='object':
        dataset_train[column]=labelencoder.fit_transform[dataset_test[column]]
    if dataset_test[column].type=='object':
        dataset_test[column]=labelencoder.fit_transform[dataset_test[column]]

dataset_train=pd.get_dummies(dataset_train)
dataset_test=pd.get_dummies(dataset_test)
    
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset_train, target_y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#check accuracies
def checkAcc(classifier,var):
    # applying k-fold cross validation
    from sklearn.model_selection import cross_val_score
    accuracies =  cross_val_score(estimator = classifier,X = X_train,y = y_train,cv = 10)
    print('accuracies mean value of ' + var, accuracies.mean())
    print('accuracies std value of ' + var, accuracies.std())
    print('\n')

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier_DT = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_DT.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier_DT.predict(X_test)
# Making the Confusion Matrix
cm_DT = confusion_matrix(y_test, y_pred)
print('Confusion Matrix of Decision Tree is: ', cm_DT)
#call check accuracies
checkAcc(classifier_DT,'Decision Tree')

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier_LR = LogisticRegression(random_state = 0)
classifier_LR.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier_LR.predict(X_test)
# Making the Confusion Matrix
cm_LR = confusion_matrix(y_test, y_pred)
print('Confusion Matrix of Decision Tree is: ', cm_LR)
#call check accuracies
checkAcc(classifier_DT,'Decision Tree')

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier_KSVC = SVC(kernel = 'rbf', random_state = 0)
classifier_KSVC.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier_KSVC.predict(X_test)
# Making the Confusion Matrix
cm_KSVC = confusion_matrix(y_test, y_pred)
print('Confusion Matrix of Decision Tree is: ', cm_KSVC)
#call check accuracies
checkAcc(classifier_DT,'Decision Tree')

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier_SVC = SVC(kernel = 'linear', random_state = 0)
classifier_SVC.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier_SVC.predict(X_test)
# Making the Confusion Matrix
cm_SVC = confusion_matrix(y_test, y_pred)
print('Confusion Matrix of Decision Tree is: ', cm_SVC)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier_NB = GaussianNB()
classifier_NB.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier_NB.predict(X_test)
# Making the Confusion Matrix
cm_NB = confusion_matrix(y_test, y_pred)
print('Confusion Matrix of Decision Tree is: ', cm_NB)


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_RF.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier_RF.predict(X_test)
# Making the Confusion Matrix
cm_RF = confusion_matrix(y_test, y_pred)
print('Confusion Matrix of Decision Tree is: ', cm_RF)

# Fitting Classification Model to the Training set
from xgboost import XGBClassifier
classifier_XGB = XGBClassifier(n_estimators=100, learning_rate=0.1, subsample=0.5, max_depth=8, min_child_weight=3)
classifier_XGB.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier_XGB.predict(X_test)
# Making the Confusion Matrix
cm_XGB = confusion_matrix(y_test, y_pred)
print('Confusion Matrix of xgboost is: ', cm_XGB)

#Output CSV
output_csv = pd.DataFrame({'SK_ID_CURR':test_id, 'TARGET':y_pred})
output_csv.to_csv('Prediction.csv', index=False)


































