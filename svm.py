import pandas as pd
import numpy as np
import os

from sklearn.metrics import accuracy_score, classification_report, multilabel_confusion_matrix, recall_score, f1_score, balanced_accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import LabelEncoder,  MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


df = pd.read_csv('df.csv').dropna()

df = df[df.irr > 100]


### DATASET TEVE DE SER REDUZIDO PARA SER COMPUTCIONALMENTE MAIS LEVE ###

print()
print('df_values', df.value_counts())

#Determination of the predictors and the criterion

columns_x=['irr','pvt','idc1','idc2','vcd1','vdc2']
columns_y=['f_nv']
x = df[columns_x]
y = df[columns_y]



#Balancing dataset

over = SMOTE()
x,y = over.fit_resample(x,y)


## #Normalization fo the values ###

sc = MinMaxScaler()
x = sc.fit_transform(x)



x = pd.DataFrame(x)
y = pd.DataFrame(y)

print()
print('\n y_values',  y.value_counts())
print()
print('------------------------------------------------------------------------')
print()
print(' \n x_values',  x.value_counts())
print()
print('------------------------------------------------------------------------')

### Train-Validation-Test Split ###

x_train, x_test, y_train, y_test = train_test_split(x, y,train_size = 0.8, test_size=0.2)

x_train = pd.DataFrame(x_train)
y_train = pd.DataFrame(y_train)

print()
print('\n y__train_values',  y_train.value_counts())
print()

print()
print(' \n x_train_values',  x_train.value_counts())
print()

print('------------------------------------------------------------------------')

x_test = pd.DataFrame(x_test)
y_test = pd.DataFrame(y_test)

print()
print('\n y__test_values',  y_test.value_counts())
print()

print()
print(' \n x_test_values',  x_test.value_counts())
print()

SVC = svm.SVC(C=0.5, kernel='rbf', degree=6, gamma='scale', class_weight = 'balanced', decision_function_shape='ovo')                 #class_weight = 'balanced' decision_function_shape='ovo'
SVC_model = SVC.fit(x_train, y_train)



y_pred = SVC_model.predict(x_test)

print()
print('------------------------------------------------------------------------------------------------------')
print()

print('\n y_pred', y_pred)

### Evaluate model ###

print()
print('------------------------------------------------------------------------------------------------------')
print()

print('Classification Report: ', classification_report(y_test,y_pred))

# Generate multiclass confusion matriceS

cf = confusion_matrix(y_test, y_pred)
print(cf)

cf0 = pd.DataFrame(cf)
cf0.to_csv('cf_ban.csv')

#Printing to csv

y_pred = pd.DataFrame(y_pred)

encoder = OneHotEncoder()
encoded_y = encoder.fit(y_pred.values.reshape(-1,1))
y_pred = encoded_y.transform(y_pred.values.reshape(-1,1)).toarray()


y_pred = pd.DataFrame(y_pred)
y_pred.to_csv('pred_svm.csv')
