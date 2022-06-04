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
plt.style.use('fivethirtyeight')
# DATADIR = (pathlib.Path(__file__).parent.resolve())

IMG_SIZE = 200


# #Load my actual data
DATADIR = 'IMG'
CATEGORIES = ['fogao', 'geladeiras']
# CATEGORIES = ['Cat', 'Dog']
# print(os.path.join(DATADIR, 'OK'))

training_data = []

def create_training_data():
  for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    # path = DATADIR + "/" + category
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
      try:
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        training_data.append([new_array, class_num])
      except Exception as e:
        pass

create_training_data()

random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
  X.append(features)
  y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y).reshape(-1,1)


y_one = to_categorical(y)
# print(y_one)

import pickle

pickle_out = open('X.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_out = open('y_one.pickle', 'wb')
pickle.dump(y_one, pickle_out)
pickle_out.close()