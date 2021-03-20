import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import hickle as hkl

data = hkl.load('data.hkl')
X_train = data['xtrain']
X_test = data['xtest']
y_train = data['ytrain']
y_test = data['ytest']
'''data = pd.read_csv('train.csv')
X = data.iloc[:, 0:20].values
y = data.iloc[:, 20:21].values


scaler = StandardScaler()
scaler.fit(X)
X_std = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.1)'''
best = None
acc = 0
k = 0
for i in list(range(1, 31)):
    KNN_model = KNeighborsClassifier(n_neighbors=i)
    KNN_model.fit(X_train, y_train)
    KNN_score = KNN_model.score(X_test, y_test)
    if KNN_score > acc:
        acc = KNN_score
        best = KNN_model
        k = i
print(f"Najbolji model je za k = {k}, sa tacnoscu {acc}")
KNN_pred = best.predict(X_test)
KNN_pred_train = best.predict(X_train)
KNN_matrix = confusion_matrix(y_test, KNN_pred)
KNN_matrix_train = confusion_matrix(y_train, KNN_pred_train)
print("Trenirajuci skup: ",classification_report(y_train, KNN_pred_train))
print("Testirajuci skup: ",classification_report(y_test, KNN_pred))
plt.figure(figsize=(10, 7))
sns.heatmap(KNN_matrix_train, annot=True)
plt.show()
plt.figure(figsize=(10, 7))
sns.heatmap(KNN_matrix, annot=True)
plt.show()


