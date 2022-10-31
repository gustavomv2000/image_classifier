import os
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
from time import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Activation
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
import pickle
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

pickle_in = open('X.pickle', 'rb')
X = pickle.load(pickle_in)
y = pickle.load(open('y.pickle', 'rb'))
y_one = pickle.load(open('y_one.pickle', 'rb'))

X_test = pickle.load(open('X_test.pickle', 'rb'))
y_test = pickle.load(open('y_test.pickle', 'rb'))
y_one_test = pickle.load(open('y_one_test.pickle', 'rb'))

X = X/255.0
X_test = X_test/255.0

dense_layers = [1]
layer_sizes = [16]
conv_layers = [2]  
max_pooling = [2]
res = [4]
dropouts = [0]
drop_rates = [0.2]

model = Sequential()

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            for max_p in max_pooling:
                for r in res:
                    for drops in dropouts:
                        for d_rate in drop_rates:
                            NAME = '{}-conv-{}-nodes-{}-dense-{}-max_p-{}-res-{}-dropouts-{}-droprate-{}'.format(conv_layer, layer_size, dense_layer, max_p, r, drops, d_rate, int(time()))

                            print(NAME)

                            model.add(Conv2D(layer_size, (r,r), input_shape = X.shape[1:]))
                            model.add(Activation('relu'))
                            model.add(MaxPooling2D(pool_size=(max_p,max_p)))

                            for l in range(conv_layer-1):
                                model.add(Conv2D(layer_size, (r,r)))
                                model.add(Activation('relu'))
                                model.add(MaxPooling2D(pool_size=(max_p,max_p)))

                            model.add(Flatten())

                            for l in range(dense_layer):
                                model.add(Dense(layer_size))
                                model.add(Activation('relu'))

                            for l in range(drops):
                                model.add(Dropout(d_rate))

                            model.add(Dense(2))
                            model.add(Activation('softmax'))

                            tensorboard = TensorBoard(log_dir='logs\\{}'.format(NAME))
                            

                            model.compile(loss="binary_crossentropy",
                                        optimizer="adam",
                                        metrics=["accuracy"])

                            # model.fit(X, y_one, batch_size=32, epochs=20, validation_split=0.3, callbacks=[tensorboard])

hist = model.fit(X, y_one, batch_size=32, epochs=20, validation_split=0.3)

# results = model.evaluate(X_test, y_test, batch_size=32)
# print(results)

# model.save('models/mymodel')

results = model.evaluate(X_test, y_one_test)[1]
print(results)

#Visualize the models accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'upper left')
plt.show()

#Visualize the models loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'upper right')
plt.show()

model.save('models/mymodel')