import tensorflow as ts
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use('fivethirtyeight')

model = keras.models.load_model('models/mymodel')

DATADIR = 'C:\Projects\image_classifier\classifier\IMG'
CATEGORIES = ['OK_TEST', 'NOK_TEST']

for category in CATEGORIES:
  path = os.path.join(DATADIR, category)
  # path = DATADIR + "/" + category
  class_num = CATEGORIES.index(category)
  for img in os.listdir(path):
    try:
      new_image = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
      # new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
      # training_data.append([new_array, class_num])
    
      #Show the image
      # new_image = cv2.imread('test_images/3.png', cv2.IMREAD_GRAYSCALE)
      #timg = plt.imshow(new_image)

      #Resize the image
      resized_image = cv2.resize(new_image, (200,200))
      #img = plt.imshow(resized_image, cmap='gray')

      predictions = model.predict(np.array([resized_image]))

      #Sort the predictions from least to greatest
      list_index = [0,1] #2 different classes
      x = predictions

      for i in range(2):
        for j in range(2):
          if x[0][list_index[i]] > x[0][list_index[j]]:
            temp = list_index[i]
            list_index[i] = list_index[j]
            list_index[j] = temp

      #Show the sorted labels in order
      # print(list_index)

      #print the first 5 predictions
      classification = ['OK', 'NOK']
      for i in range(2):
        print(classification[list_index[i]], ':', round(predictions[0][list_index[i]] * 100, 2), '%')

      if category == 'OK_TEST':
        print('Expected OK')
        print('\n')
      else:
        print('Expected NOK')
        print('\n')
      
    except Exception as e:
      print('did not read: ', e)