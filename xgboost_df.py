import pandas as pd
import numpy as np
import os
import shutil
import pickle as pk
#import matplotlib as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import class_weight
import time


#Loading the data

df = pd.read_csv('df.csv').dropna()

df = df[df.irr > 100]



# Determination of the predictors and the criterion
columns_x=['irr','pvt','idc1','idc2','vcd1','vdc2']
columns_y=['f_nv']
x = df[columns_x]
y = df[columns_y]



#Balancing dataset

x=pd.DataFrame(x)
y=pd.DataFrame(y)

#print('\n x2', x.value_counts())
#print('\n y2', y.value_counts())

#Encoding

encoder = OneHotEncoder()
encoded_y = encoder.fit(y.values.reshape(-1,1))
y = encoded_y.transform(y.values.reshape(-1,1)).toarray()



#Normalization fo the values

sc = MinMaxScaler()
x = sc.fit_transform(x)

# Train-Validation-Test Split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Building model and test time

xgb_model = xgb.XGBClassifier(booster='gbtree',eta=0.05,subsample=0.05,max_depth=4,min_child_weight=5)
xgb_model.fit(x_train, y_train)
y_pred = xgb_model.predict(x_test)
start_time_1 = time.time()

end_time_1 = time.time()
elapsed_time_1 = end_time_1 - start_time_1
print("Time taken to fit the classifier: {elapsed_time_1:.2f} seconds")

start_time_2 = time.time()

end_time_2 = time.time()
elapsed_time_2 = end_time_2 - start_time_2
print("Time taken tfor the classifier to make predictions: {elapsed_time_2:.2f} seconds")

print()










print()
#print('scores')
print(cross_val_score(xgb_model, x_train, y_train, scoring='accuracy', cv=5))



print()
print('accuracy : ', accuracy_score(y_pred,y_test))

print()
#print('accuracy : ', precision_recall_fscore_support(y_pred,y_test)*100)




y_pred0 = np.argmax(y_pred, axis=1)
y_test0 = np.argmax(y_test, axis=1)
cf= confusion_matrix(y_pred0,y_test0)

print()
print('confusion matrix')
print(cf)

print('--------------------------------------------------------------------------------------------------')
print()
print('Classification Report')
print(classification_report(y_pred,y_test))
