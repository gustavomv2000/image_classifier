import tensorflow as ts
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Activation
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import pickle
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

model = Sequential()

model.add(Conv2D(100, (5,5), input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

#model.add(Dense(500, activation="relu"))

#model.add(Dropout(0.5))

#model.add(Dense(250, activation = 'relu'))

model.add(Dense(2, activation='softmax'))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

hist = model.fit(X, y_one, batch_size=64, epochs=10, validation_split=0.1)

#Evaluate the model using the test data set
results = model.evaluate(X_test, y_one_test)[1]
print(results)

#Visualize the models accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
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