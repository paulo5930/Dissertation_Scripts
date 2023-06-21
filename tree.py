
import pandas as pd                                                                                                                 
import numpy as np  
from data_processing import DataProcessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler  
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, balanced_accuracy_score, classification_report, precision_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree




#Load dataset

data_proc = DataProcessing()
df = pd.read_csv('df.csv')

#Inputs and targets
feature_cols_x = ['irr','pvt','idc1','idc2','vcd1','vdc2']
feature_cols_y = ['f_nv']

df = df[df.irr >100]


x = df[feature_cols_x]
y = df[feature_cols_y]


#OneHotEncoding

y = pd.DataFrame(y)
encoder = OneHotEncoder()
encoded_y = encoder.fit(y.values.reshape(-1,1))
y = encoded_y.transform(y.values.reshape(-1,1)).toarray()

#Undersampling dataset

sc = StandardScaler()
x = sc.fit_transform(x)



#train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)



#Model Build

clf = DecisionTreeClassifier(criterion='gini', splitter = 'best',max_depth=4,max_leaf_nodes=7,class_weight='balanced')             #criterion='gini', splitter = 'best', max_depth=3, max_leaf_nodes=6
dtree = clf.fit(x_train, y_train)
y_pred = dtree.predict(x_test)


# Decison Tree plotting

fig = plt.figure(figsize=(25,20))
tree.plot_tree(clf, class_names = True, filled=False)
fig.savefig("decistion_tree.png")

#Evaluating

print()
print('----------------------------------------------------------------------------------------------------------------------------------------------------------------')
print()

print('\n accuracy', accuracy_score(y_test, y_pred))
print('\n recall', recall_score(y_test, y_pred, average='macro'))
print('\n f1 score', f1_score(y_test, y_pred, average='macro'))
print('\n precison', precision_score(y_test, y_pred, average='macro'))

print()
print('y_test', y_test)
print()
print('y_pred', y_pred)




y_pred=np.argmax(y_pred, axis=1)
y_test=np.argmax(y_test, axis=1)



cf =confusion_matrix(y_test,y_pred)
#f = confusion_matrix(y_test, y_pred_multilabel)
print('\n cf')
print('\n', cf)

cf0 = pd.DataFrame(cf)
cf0.to_csv('cf_unb.csv')

print()
print('----------------------------------------------------------------------------------------------------------------------------------------------------------------')
print()

print(' Balanced_accuracy_test:', balanced_accuracy_score(y_test, y_pred))

print()
print('----------------------------------------------------------------------------------------------------------------------------------------------------------------')
print()

print('Accuracy_score: {:.2f}%', accuracy_score(y_test, y_pred)*100)
print()



print('Classification Report')
print(classification_report(y_pred,y_test))


#Second Encoding

y_pred =pd.DataFrame(y_pred)
y_test =pd.DataFrame(y_test)

encoder = OneHotEncoder()
encoded_y = encoder.fit(y_pred.values.reshape(-1,1))
y_pred = encoded_y.transform(y_pred.values(-1,1)).toarray()

y_pred_csv =pd.DataFrame(y_pred).round(decimals=0)
#y_pred_csv.to_csv('y_pred.csv')
