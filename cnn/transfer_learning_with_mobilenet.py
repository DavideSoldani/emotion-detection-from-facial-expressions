import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
import time
import pickle
from sklearn.metrics import classification_report
from keras.optimizers import RMSprop

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

#input deve essere almeno 128
base_model = MobileNet(weights='imagenet',include_top=False, input_shape=(128,128,3)) #imports the mobilenet model and discards the last 1000 neuron layer.

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x = Dense(1024,activation='relu')(x) #dense layer 2
x = Dense(512,activation='relu')(x) #dense layer 3
preds = Dense(7,activation='softmax')(x) #final layer with softmax activation

model = Model(inputs=base_model.input,outputs=preds)

#print layers names
for i,layer in enumerate(model.layers):
	print(i,layer.name)

#lock training for mobilenet layers
for layer in model.layers[:86]:
    layer.trainable=False
for layer in model.layers[87:]:
	layer.trainable=True

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

optimizer = keras.optimizers.SGD(lr=0.01, decay=0.1, momentum=0.975, nesterov=False)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=3, validation_split=0.3, callbacks=[tensorboard])

model.save('emotion_cnn_transfer.model')


# get the predictions for the test data
predicted_classes = model.predict(X_test)
predicted_classes = predicted_classes.argmax(axis=-1)
predicted_classes = keras.utils.to_categorical(predicted_classes, num_classes)


# get the indices to be plotted
correct = np.where(predicted_classes==y_test)[0]
incorrect = np.where(predicted_classes!=y_test)[0]

target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_test, predicted_classes, target_names=target_names))


