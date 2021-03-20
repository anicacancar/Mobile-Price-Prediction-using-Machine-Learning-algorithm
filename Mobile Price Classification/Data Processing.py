import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("train.csv")
X = data.drop('price_range', axis=1)
y = data['price_range']

# Standardizacija
scaler = StandardScaler()
scaler.fit(X)
X_std = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.1)

import hickle as hkl
data = {'xtrain': X_train, 'xtest': X_test,'ytrain': y_train,'ytest':y_test}
hkl.dump(data,'data.hkl')
