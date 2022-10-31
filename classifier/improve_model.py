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


config = ConfigProto()
#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
config.gpu_options.allow_growth = True
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
sess = tf.Session(config=config)

pickle_in = open('X.pickle', 'rb')
X = pickle.load(pickle_in)
y = pickle.load(open('y.pickle', 'rb'))
y_one = pickle.load(open('y_one.pickle', 'rb'))

X = X/255.0

pickle_in = open('X_test.pickle', 'rb')
X_test = pickle.load(pickle_in)
y_test = pickle.load(open('y_test.pickle', 'rb'))
y_one_test = pickle.load(open('y_one_test.pickle', 'rb'))

X_test = X_test/255.0

dense_layers = [1]
layer_sizes = [8]
conv_layers = [1, 2]  
max_pooling = [2]
res = [4]
dropouts = [0, 1]
drop_rates = [0.4]

results_acc = []
results_loss = []

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            for max_p in max_pooling:
                for r in res:
                    for drops in dropouts:
                        for d_rate in drop_rates:
                            NAME = '{}-conv-{}-nodes-{}-dense-{}-max_p-{}-res-{}-dropouts-{}-droprate-{}'.format(conv_layer, layer_size, dense_layer, max_p, r, drops, d_rate, int(time()))

                            print(NAME)
                            model = Sequential()

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

                            if(dense_layer != 0):
                                for l in range(drops):
                                    model.add(Dropout(d_rate))

                            model.add(Dense(2))
                            model.add(Activation('softmax'))

                            tensorboard = TensorBoard(log_dir='logs\\{}'.format(NAME))
                            

                            model.compile(loss="binary_crossentropy",
                                        optimizer="adam",
                                        metrics=["accuracy"])

                            model.fit(X, y_one, batch_size=32, epochs=20, validation_split=0.3, callbacks=[tensorboard])
                            score = model.evaluate(X_test, y_one_test, verbose = 0) 
                            results_acc.append(score[1])
                            results_loss.append(score[0])
                            print('results acc: ', score[1])
                            print('results loss: ', score[0])

print('results acc: ', results_acc)
print('results loss: ', results_loss)