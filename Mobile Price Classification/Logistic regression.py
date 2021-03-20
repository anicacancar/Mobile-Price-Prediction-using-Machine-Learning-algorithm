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
import hickle as hkl

data = hkl.load('data.hkl')
X_train = data['xtrain']
X_test = data['xtest']
y_train = data['ytrain']
y_test = data['ytest']

'''# Standardizacija
scaler = StandardScaler()
scaler.fit(X)
X_std = scaler.transform(X)
# Podjela podatka
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.1)'''

LR_model = LogisticRegression()
LR_model.fit(X_train, y_train)
LR_pred = LR_model.predict(X_test)
LR_pred_tren = LR_model.predict(X_train)
LR_score = LR_model.score(X_test, y_test)
print(f"Nakon standardizacije {LR_score}")

LR_score_train = LR_model.score(X_train, y_train)
LR_matrix = confusion_matrix(y_test, LR_pred)
LR_matrix_train = confusion_matrix(y_train, LR_pred_tren)
print(classification_report(y_train, LR_pred_tren))
print(classification_report(y_test, LR_pred))

plt.figure(figsize=(10, 7))
sns.heatmap(LR_matrix_train, annot=True)
plt.show()
plt.figure(figsize=(10, 7))
sns.heatmap(LR_matrix, annot=True)
plt.show()

