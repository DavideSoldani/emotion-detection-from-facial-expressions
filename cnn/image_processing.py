import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import pickle
import random
import keras
from keras import regularizers

NAME = "Emotion-CNN"

TRAINDIR = "C:/Users/Davide Soldani/training"
TESTDIR = "C:/Users/Davide Soldani/test"

CATEGORIES = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

num_classes = 7

IMG_SIZE = 224

training_data = []
test_data = []

def create_training_data():
    for category in CATEGORIES:  
        path = os.path.join(TRAINDIR,category)  
        class_num = CATEGORIES.index(category)  

        for img in tqdm(os.listdir(path)): 
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)  # convert to array #IMREAD_COLOR #IMREAD_GRAYSCALE
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass

create_training_data()

def create_test_data():
    for category in CATEGORIES:  
        path = os.path.join(TESTDIR,category)  
        class_num = CATEGORIES.index(category)  

        for img in tqdm(os.listdir(path)):  
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                test_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass

create_test_data()

random.shuffle(training_data)
random.shuffle(test_data)

X_train = []
y_train = []

X_test = []
y_test = []

for features,label in training_data:
    X_train.append(features)
    y_train.append(label)

X_train = np.array(X_train).reshape(-3, IMG_SIZE, IMG_SIZE, 3)
y_train = keras.utils.to_categorical(y_train, num_classes)

for features,label in test_data:
    X_test.append(features)
    y_test.append(label)

X_test = np.array(X_test).reshape(-3, IMG_SIZE, IMG_SIZE, 3)
y_test = keras.utils.to_categorical(y_test, num_classes)


pickle_out = open("X_train.pickle","wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()

pickle_out = open("y_train.pickle","wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()

pickle_out = open("X_test.pickle","wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle","wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()

