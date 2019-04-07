# -*- coding: utf-8 -*-
"""
Let's train a thing
"""

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense


iris = load_iris()

iris_spec = to_categorical(iris.target)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris_spec,
                                                    test_size=0.2)

# Scale the data fairly
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = Sequential()
model.add(Dense(units=10, input_shape=(4,)))
model.add(Dense(units=3, activation="softmax"))

model.compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = ['accuracy'])
  
history = model.fit(X_train_s, 
                    y_train, 
                    epochs = 100, 
                    validation_data = (X_test_s, y_test))

model.save("iris_model_python.hdf5")
