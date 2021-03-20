import numpy as np
import pandas as pd
import sklearn
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler,normalize, OneHotEncoder
from sklearn.model_selection import train_test_split
from kerastuner import RandomSearch
from sklearn.metrics import confusion_matrix, classification_report
import time
from matplotlib import pyplot as plt
import seaborn as sns

data = pd.read_csv("train.csv")
X = data.iloc[:, 0:20].values
y = data.iloc[:, 20:21].values

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

ohe = OneHotEncoder()
ohe.fit(y)
y = ohe.transform(y).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

def build_model(hp):
    model = keras.models.Sequential()

    model.add(Dense(hp.Int("input units", min_value=8, max_value=32, step=2), input_dim=20,
                    activation=hp.Choice('dense_activation', values=['relu', 'tanh', 'sigmoid'], default='relu')))
    for i in range(hp.Int("n_layers", 1, 4)):
        model.add(Dense(hp.Int("input units", min_value=8, max_value=32, step=2), input_dim=20,
                        activation=hp.Choice('dense_activation', values=['relu', 'tanh', 'sigmoid'], default='relu')))
        model.add(
            Dropout(
                rate=hp.Float(
                    'dropout_3',
                    min_value=0.0,
                    max_value=0.5,
                    default=0.25,
                    step=0.05
                )
            )
        )
        model.add(Dense(4, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


LOG_DIR = f"{int(time.time())}"

tuner = RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=1,
    executions_per_trial=1,
    directory=LOG_DIR
)

tuner.search(x=X_train, y=y_train, epochs=150, batch_size=64, validation_data=(X_test, y_test))
print(tuner.get_best_hyperparameters()[0].values)
print(tuner.results_summary())

best_model = tuner.get_best_models(num_models=1)[0]
NN_pred = best_model.predict(X_test)
pred = list()
for i in range(len(NN_pred)):
    pred.append(np.argmax(NN_pred[i]))
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))
train_pred = best_model.predict(X_train)
train_prediction = list()
for i in range(len(train_pred)):
    train_prediction.append(np.argmax(train_pred[i]))
train = list()
for i in range(len(y_train)):
    train.append(np.argmax(y_train[i]))


NN_matrix = confusion_matrix(test, pred)
print("Testirajući skup: ", classification_report(test, pred))
NN_matrix_train = confusion_matrix(train, train_prediction)
print("Trenirajući skup: ", classification_report(train, train_prediction))

plt.figure(num=1, figsize=(10, 7))
sns.heatmap(NN_matrix, annot=True)
plt.show()

plt.figure(num=2, figsize=(10, 7))
sns.heatmap(NN_matrix_train, annot=True)
plt.show()

