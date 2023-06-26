import pandas as pd
import numpy as np
import os
import shutil
import pickle as pk
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, f1_score, balanced_accuracy_score, precision_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

#Loading the data

df = pd.read_csv('df.csv').dropna()

df = df[df.irr > 100]

# Determination of the predictors and the criterion
columns_x=['irr','pvt','idc1','idc2','vcd1','vdc2']
columns_y=['f_nv']#
x = df[columns_x]
y = df[columns_y]

print()
print('Data Strucuture (5 rows)')
print()
print(df.head())



x=pd.DataFrame(x)
y=pd.DataFrame(y)

print('\n x2', x.value_counts())
print('\n y2', y.value_counts())

#One Hot Encoding

encoder = OneHotEncoder()
encoded_y = encoder.fit(y.values.reshape(-1,1))
y = encoded_y.transform(y.values.reshape(-1,1)).toarray()

#Normalization ofthe values

sc = StandardScaler()
x = sc.fit_transform(x)


# Train-Test-Split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


#Model Structure

model = MLPClassifier(hidden_layer_sizes=(3,13), activation='relu', solver = 'adam',learning_rate='adaptive', max_iter = 30)
MLP = model.fit(x_train ,y_train)
y_pred = MLP.predict(x_test)



# Metrics 

y_pred=np.argmax(y_pred, axis=1)
y_test=np.argmax(y_test, axis=1)

print('\n accuracy', accuracy_score(y_test, y_pred))
print('\n recall', recall_score(y_test, y_pred, average='macro'))
print('\n f1 score', f1_score(y_test, y_pred, average='macro'))
print('\n precison', precision_score(y_test, y_pred, average='macro'))

print()
print('---------------------------------------------------------------------------------------------------------------------------------------------------------------------')
print()

cf = confusion_matrix(y_test, y_pred)
print('\n cf')
print( cf)


cf0 = pd.DataFrame(cf)
cf0.to_csv('cf_unb.csv')
print()
print('----------------------------------------------------------------------------------------------------------------------------------------------------------------')
print()

print(' Balanced accuracy:', balanced_accuracy_score(y_pred, y_test))

#2º Encoding and Print predictions to csv

y_pred = pd.DataFrame(y_pred).round(decimals=0)

print('---------------------------------------------------------------------------------------------------------------------------------------------------------------------')
print()
print('Classification Report: ')
print(classification_report(y_pred, y_test))
encoder = OneHotEncoder()
encoded_y = encoder.fit(y_pred.values.reshape(-1,1))
y_pred = encoded_y.transform(y_pred.values.reshape(-1,1)).toarray()

#pred = pd.DataFrame(y_pred).round(decimals=0)
#pred.to_csv('maisumpred.csv')


