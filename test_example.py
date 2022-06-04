import tensorflow as ts
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use('fivethirtyeight')

model = keras.models.load_model('models/mymodel')

DATADIR = 'IMG'
CATEGORIES = ['fogao_test', 'geladeira_test']

for category in CATEGORIES:
  path = os.path.join(DATADIR, category)
  class_num = CATEGORIES.index(category)
  for img in os.listdir(path):
    # try:
    new_image = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
    
    resized_image = cv2.resize(new_image, (100,100))
    X = resized_image
    
    resized_image = np.array(X).reshape(100, 100, 1)

    predictions = model.predict(np.array([resized_image]))

    list_index = [0,1] #2 different classes
    x = predictions

    for i in range(2):
      for j in range(2):
        if x[0][list_index[i]] > x[0][list_index[j]]:
          temp = list_index[i]
          list_index[i] = list_index[j]
          list_index[j] = temp

    classification = ['fogao_test', 'geladeira_test']
    for i in range(2):
      print(classification[list_index[i]], ':', round(predictions[0][list_index[i]] * 100, 2), '%')

    if category == 'fogao_test':
      print('Expected fogao_test')
      print('\n')
    else:
      print('Expected geladeira_test')
      print('\n')