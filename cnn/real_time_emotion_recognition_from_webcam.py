import cv2
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

model = tf.keras.models.load_model('emotion_cnn_transfer-vgg.model') #model created by transfer_learning_with_vggface.py

CATEGORIES = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]


cap = cv2.VideoCapture(0)


font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (30,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

while True:
	
	x = []
	ret, image_np = cap.read()
	image = cv2.resize(image_np,(800,600))
	cv2.imshow('image', image)

	
	new_array = cv2.resize(image_np, (224, 224)) 
	x.append(new_array)
	x = np.array(x).reshape(-3, 224, 224, 3)
	x = x/255.0
	predicted_class = model.predict(x).argmax(axis=-1)
	predicted_class = CATEGORIES[predicted_class.item(0)]
	cv2.putText(image, predicted_class, bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
	cv2.imshow('image', image)
	print(predicted_class)
	

	if cv2.waitKey(25) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		cap.release()
		break

