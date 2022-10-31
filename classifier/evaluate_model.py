#libs
import tensorflow as ts
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Activation
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

import os
import cv2
import random
import pathlib
import pickle
import math
plt.style.use('fivethirtyeight')

model = keras.models.load_model('models/mymodel')

pickle_in = open('X_test.pickle', 'rb')
X_test = pickle.load(pickle_in)
y_test = pickle.load(open('y_test.pickle', 'rb'))
y_one_test = pickle.load(open('y_one_test.pickle', 'rb'))
X_test = X_test/255.0

score = model.evaluate(X_test, y_one_test, verbose = 0) 

print('Test loss test data:', score[0]) 
print('Test accuracy test data:', score[1])

pickle_in = open('X.pickle', 'rb')
X = pickle.load(pickle_in)
y = pickle.load(open('y.pickle', 'rb'))
y_one = pickle.load(open('y_one.pickle', 'rb'))
X = X/255.0

score = model.evaluate(X, y_one, verbose = 0) 

print('Test loss:', score[0]) 
print('Test accuracy:', score[1])


