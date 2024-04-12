import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
x_train = pd.read_csv("x_train.csv")
x_test = pd.read_csv("x_test.csv")
y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
scaler = StandardScaler()
x_trainscaled = scaler.fit_transform(x_train)
x_testscaled = scaler.fit_transform(x_test)
print(x_testscaled)
print(x_trainscaled)
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
cnn_4_layer = Sequential()
cnn_4_layer.add(Dense(128, activation='relu', input_shape=(x_trainscaled.shape[1],)))
cnn_4_layer.add(Dense(64, activation='relu'))
cnn_4_layer.add(Dense(32, activation='relu'))
cnn_4_layer.add(Dense(16, activation='relu'))
cnn_4_layer.add(Dense(1, activation='sigmoid'))

cnn_4_layer.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cnn_4_layer.fit(x_trainscaled, y_train, epochs=10, batch_size=32, validation_data=(x_testscaled, y_test))

loss, accuracy = cnn_4_layer.evaluate(x_testscaled, y_test)
print(f'Accuracy on the test set: {accuracy}')
cnn4_predictions = cnn_4_layer.predict(x_trainscaled)
cnn4_predictions_test = cnn_4_layer.predict(x_testscaled)

lrcnn_model = LogisticRegression()
#print(x_train2)
lrcnn_model.fit(cnn4_predictions, y_train)
#joblib.dump(lr_model, 'logistic_regression_2_2.pkl')

lr_test_predictions = lrcnn_model.predict(cnn4_predictions_test)



accuracy = accuracy_score(y_test, lr_test_predictions)


print(f"Accuracy: {accuracy}")
cnn_4_layer.save('cnn_logistic.keras')
joblib.dump(lrcnn_model, 'cnn_logistic_regression.pkl')