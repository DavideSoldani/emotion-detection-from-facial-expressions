import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time
import keras
from keras import regularizers
import pickle
from sklearn.metrics import classification_report

NAME = "Emotion-CNN"

num_classes = 7

pickle_in = open("X_train.pickle","rb")
X_train = pickle.load(pickle_in)

pickle_in = open("y_train.pickle","rb")
y_train = pickle.load(pickle_in)

pickle_in = open("X_test.pickle","rb")
X_test = pickle.load(pickle_in)

pickle_in = open("y_test.pickle","rb")
y_test = pickle.load(pickle_in)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train/255.0
X_test = X_test/255.0

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  

model.add(Dense(64))
model.add(Dropout(0.3))
model.add(Dense(16))
model.add(Dense(num_classes, activation='softmax'))

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

#optimizer = keras.optimizers.Adam(lr=0.01, decay=0.1, momentum=0.975, nesterov=False)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=3, validation_split=0.3, callbacks=[tensorboard])

model.save('emotion_cnn.model')


# get the predictions for the test data
predicted_classes = model.predict_classes(X_test)
predicted_classes = keras.utils.to_categorical(predicted_classes, num_classes)
# get the indices to be plotted
correct = np.where(predicted_classes==y_test)[0]
incorrect = np.where(predicted_classes!=y_test)[0]

target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_test, predicted_classes, target_names=target_names))





