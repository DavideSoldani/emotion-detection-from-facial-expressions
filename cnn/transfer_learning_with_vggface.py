from keras.engine import  Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense,GlobalAveragePooling2D, Flatten
from tensorflow.keras.callbacks import TensorBoard
import time
import pickle
from sklearn.metrics import classification_report
from keras.optimizers import RMSprop
from keras import regularizers


NAME = "Emotion-CNN-VGG"

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


#input have to be at least 128
base_model = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg') 

x = base_model.get_layer('pool5').output
x = Flatten()(x)
x = Dense(1024,activation='relu')(x)
x = Dense(1024,activation='relu')(x)
x = Dense(512,activation='relu')(x)
preds = Dense(7,activation='softmax')(x) #final layer with softmax activation

model = Model(inputs=base_model.input,outputs=preds)


for layer in model.layers[:18]:
    layer.trainable=False
for layer in model.layers[19:]:
	layer.trainable=True


tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=3, validation_split=0.3, callbacks=[tensorboard])

model.save('emotion_cnn_transfer-vgg.model')


# get the predictions for the test data
predicted_classes = model.predict(X_test)
predicted_classes = predicted_classes.argmax(axis=-1)
predicted_classes = keras.utils.to_categorical(predicted_classes, num_classes)


# get the indices to be plotted
correct = np.where(predicted_classes==y_test)[0]
incorrect = np.where(predicted_classes!=y_test)[0]

target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_test, predicted_classes, target_names=target_names))
