import pandas as pd
import numpy as np
import os
import shutil
import pickle as pk
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight
import os
import lightgbm as lgb
import time


os.environ["OMP_NUM_THREADS"] = "4"


#Loading the data

df = pd.read_csv('df.csv').dropna()

df = df[df.irr > 100]


# Determination of the predictors and the criterion
columns_x=['irr','pvt','idc1','idc2','vcd1','vdc2']
columns_y=['f_nv']
x = df[columns_x]
y = df[columns_y]

print()
print('y', y.value_counts())


#Balancing dataset



#Class Weights

encoder = LabelEncoder()
y = encoder.fit_transform(y)

class_weights = compute_class_weight( class_weight = 'balanced' , classes =np.unique(y) ,y = y)
cw = dict(enumerate(class_weights))


print()
print('class_weights')
print(class_weights)

#Encoding

y = pd.DataFrame(y)
encoder = OneHotEncoder()
encoded_y = encoder.fit(y.values.reshape(-1,1))
y = encoded_y.transform(y.values.reshape(-1,1)).toarray()



#Normalization of the values

sc = StandardScaler()
x = sc.fit_transform(x)

# Train Test Split

x_train, x_test, y_train, y_test = train_test_split(x, y,train_size= 0.8, test_size=0.2)



#Building model

model = lgb.LGBMClassifier(boosting_type='gbdt',objective ='multiclass',class_weight=cw,max_depth=4,num_leaves=7)
start_time = time.time()
model.fit(x_train, y_train)

end_time = time.time()
y_pred = model.predict(x_test)

elapsed_time = end_time - start_time

print("Time taken to fit the classifier:  seconds")
print(elapsed_time )



print()
print('y_pred')
print(y_pred)

#Metrics

print()
print('scores_ttest')
print()
print('------------------------------------------------------------------------------------------------------')
print('\n accuracy', accuracy_score(y_test, y_pred))
print('\n recall', recall_score(y_test, y_pred, average='macro'))
print('\n f1 score', f1_score(y_test, y_pred, average='macro'))
print('\n precison', precision_score(y_test, y_pred, average='macro'))
print()
print()


print()
#print('accuracy_test : ', accuracy_score(y_pred,y_test)*100)
y_predd = pd.DataFrame(y_pred)

print()
print('y_pred')
print(y_pred)
print()
label = LabelEncoder()

y_predd= label.fit_transform(y_pred)

y_predd = pd.DataFrame(y_pred)

print()
print('y_preddd')
print(y_predd)
#y_predd.to_csv('y_preddd.csv')


cf= confusion_matrix(y_pred,y_test)

print()
print('confusion matrix')
print(cf)

#cf0 = pd.DataFrame(cf)

#cf0.to_csv('cf_ban.csv')
print('------------------------------------------------------------------------')
print()
print('Classification Report')
print(classification_report(y_pred,y_test))

