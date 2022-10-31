import tensorflow as ts
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('fivethirtyeight')

model = keras.models.load_model('models/mymodel')

for n in range(4):
    #Show the image
    new_image = cv2.imread('test_images/'+str(n+1)+'.png', cv2.IMREAD_GRAYSCALE)
    #timg = plt.imshow(new_image)

    #Resize the image
    resized_image = cv2.resize(new_image, (50,50))
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
    classification = ['CAT', 'DOG']
    for i in range(2):
        print(classification[list_index[i]], ':', round(predictions[0][list_index[i]] * 100, 2), '%')
    
    print('\n')