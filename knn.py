import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour
from data_processing import DataProcessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, classification_report
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

#Load Dataset

data_proc=DataProcessing()

df_amb = data_proc.matlab_to_pandas_amb()
df_elec = data_proc.matlab_to_pandas_elec()

x_amb,y = data_proc.preprocessing_amb(df_amb)
x_elec  = data_proc.preprocessing_elec(df_elec)

df = pd.concat([x_amb,x_elec,y], axis=1)

df = df[df.irr > 2]

remove_n = 580000
drop_indices = np.random.choice(df.index, remove_n, replace=False)
df = df.drop(drop_indices)

print()
print(' \n df_values', df.value_counts())

#df.to_csv('df.csv')

#Defenition of predictor sand others values

columns_x = ['irr','pvt','idc1','idc2','vcd1','vdc2']
columns_y = ['f_nv']
x = df[columns_x]
y = df[columns_y] 

#Balancing dataset

#under = RandomUnderSampler(sampling_strategy='all')
#x,y = under.fit_resample(x,y)


x = pd.DataFrame(x)
#print('\n x', x.value_counts())
y = pd.DataFrame(y)
#print('\n y', y.value_counts())

#Encoding

encoder = OneHotEncoder()
encoded_y = encoder.fit(y.values.reshape(-1,1))
y = encoded_y.transform(y.values.reshape(-1,1)).toarray()

#Normalization of values

Mm = MinMaxScaler()
x = Mm.fit_transform(x)

# Train_Validation-Test Split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


x_train = pd.DataFrame(x_train)
y_train = pd.DataFrame(y_train)

print()
print('\n\n y__train_values',  y_train.value_counts())
print()

print()
print(' \n\n x_train_values',  x_train.value_counts())
print()

print('------------------------------------------------------------------------')

x_test = pd.DataFrame(x_test)
y_test = pd.DataFrame(y_test)

print()
print('\n\n y__test_values',  y_test.value_counts())
print()

print()
print(' \n\n x_test_values',  x_test.value_counts())
print()

#under=RandomUnderSampler()
#x_train, y_train = under.fit_resample(x_train, y_train)



#i=1
#while i < 50:
    #knn = KNeighborsClassifier(n_neighbors=i)
    #print( '\n i = ', i)
    #knn.fit(x_train, y_train)
    #y_pred = knn.predict(x_test).round(decimals=0)
    #accuracy_test = accuracy_score(y_test, y_pred)
    #print("Accuracy_test: {:.2f}%".format(accuracy_test * 100))

    #print('Recall: ', recall_score(y_test,y_pred, average='macro'))
    #print('F1 Score: ', f1_score(y_test,y_pred, average = 'macro'))

    
    #i=i+1

#KNN Classifier

knn = KNeighborsClassifier(n_neighbors=1000)         #k == n_neighbors

#Train the model

knn.fit(x_train, y_train)

print()
print('\n\nKNN Score Train: ', cross_val_score(knn, x_train, y_train, cv=5))

#Predict

y_pred = knn.predict(x_test).round(decimals=0)

#Accuracy
print()
print('--------------------------------------------------------------------------------------------------------')
print()

print('\n Classification Report Test: ', classification_report(y_test, y_pred))

print()
print('--------------------------------------------------------------------------------------------------------')
print()

print(' \n\n KNN Score: ', cross_val_score(knn, y_test, y_pred, cv=5))

#pred = pd.DataFrame(y_test).round(decimals=0)
#pred.to_csv('knn_pred.csv')

#One hot encoding and evaluating

#y_test =pd.DataFrame(y_test).round(decimals=0)
#y_pred = pd.DataFrame(y_pred).round(decimals=0)

print()
print('y_test: ')
print( y_test)
print()
print('y_pred: ')
print( y_pred)
print()

#y_pred=np.argmax(y_pred, axixs=1).round(decimals=0)
#y_test=np.argmax(y_test, axis=1).round(decimals=0)

print()
print('--------------------------------------------------------------------------------------------------------')
print()
print('Recall: ')
print( recall_score(y_test,y_pred, average='macro'))
print(' F1 Score: ')
print( f1_score(y_test,y_pred, average = 'macro'))


#encoder_2 = OneHotEncoder()

#encoded_y_test = encoder_2.fit(y_test.values.reshape(-1,1))
#y_test = encoded_y_test.transform(y_test.values.reshape(-1,1)).toarray()

#encoded_y_pred = encoder_2.fit(y_pred.values.reshape(-1,1))
#y_pred = encoded_y_pred.transform(y_pred.values.reshape(-1,1)).toarray()

y_pred = pd.DataFrame(y_pred)

y_pred=np.argmax(y_pred, axis=1)
y_test=np.argmax(y_test, axis=1)

print()
print('y_test 2: ')
print( y_test)
print()
print('y_pred_2: ')
print( y_pred)
print()


cf = confusion_matrix(y_test, y_pred)
print('\n cf')
print('\n',cf)

#reverse_encoding_y_test = np.argmax(y_test, axis=1)
#print('\n gh', reverse_encoding_y_test)
#reverse_encoding_y_pred = np.argmax(y_test_pred, axis=1)
#print('\n ghf', reverse_encoding_y_pred)

#Compare

#r_e_y_test=pd.DataFrame(reverse_encoding_y_test)
#r_e_y_pred=pd.DataFrame(reverse_encoding_y_pred)
#comp = r_e_y_pred.compare(r_e_y_test, align_axis=1)
#print('\n', comp.value_counts())

