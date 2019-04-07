# -*- coding: utf-8 -*-
"""
Let's train a thing
"""
import datetime
import os
import subprocess
import sys

#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

BUCKET_NAME = 'keras-235720'

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

model_filename = "iris_model_python.hdf5"
model.save(model_filename)

gcs_model_path = os.path.join('gs://', BUCKET_NAME,
    datetime.datetime.now().strftime('iris_%Y%m%d_%H%M%S'), model_filename)
subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path],
    stderr=sys.stdout)
