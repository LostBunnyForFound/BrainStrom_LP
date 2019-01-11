#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 19:31:58 2018

@author: loretta
"""

from __future__ import print_function

import tensorflow as tf
import keras
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np
import pandas as pd

# read file
batch_size = 128
num_classes = 2
epochs = 35
def load_dataset(filename):
    pairs = []
    with tf.gfile.GFile(filename, "r") as f:
#         next(f)
        for line in f:
            ts = line.strip().split("\t")
            pairs.append(((float(ts[0]), float(ts[1]),float(ts[2]),int(ts[3]))))
    return pd.DataFrame(pairs, columns=["feature1", "feature2","feature3","label"])

#load features file
dataset = load_dataset("capture_feature.txt")
fv = dataset[["feature1","feature2","feature3"]]
lv = dataset['label']
NUM_FEATURES = fv.shape[1]
NUM_SAMPLES = fv.shape[0]
NUM_CLASSES = 2
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(fv, lv, test_size=0.33, random_state=42)

def load_test(filename):
    pairs = []
    with tf.gfile.GFile(filename, "r") as f:
#         next(f)
        for line in f:
            ts = line.strip().split("\t")
            pairs.append((float(ts[0]), float(ts[1]),float(ts[2])))
    return pd.DataFrame(pairs, columns=["feature1","feature2","feature3"])


testdata = load_test("test_feature.txt")
X_for_test = testdata[["feature1", "feature2","feature3"]]
# the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape the input
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# build keras sequential network
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(NUM_FEATURES,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# save model prediction
classes = model.predict(X_for_test, batch_size=128)
res = [value[1] for value in classes]
index = np.arange(1,len(testdata)+1)
result = pd.DataFrame({'Id': index, 'Prediction': res})
result.to_csv('test3.csv', index=False)
