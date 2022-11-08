import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

DIRECTORY = r"E:\My Files\Reyad\Study\Coding\Python\Img_Classification_CNN\leaf" # path of datasets folder
CATAGORIES = ['Strawberry_fresh', 'Strawberry_scrotch'] # datasets folder name

data = []

for catagory in CATAGORIES:
    folder = os.path.join(DIRECTORY, catagory) # connect the folder to the path
    # print(folder)
    label = CATAGORIES.index(catagory)

    for img in os.listdir(folder):
        img = os.path.join(folder, img) # join path for every images
        img_display = cv2.imread(img) # image reading with cv2 library
        img_display = cv2.resize(img_display, (100, 100)) # resize image : 100 x 100
        # plt.imshow(img_display)
        # plt.show()
        data.append([img_display, label])
        # print(data)

random.shuffle(data)

x = [] # for images
y = [] # for label

for features, label in data:
    x.append(features)
    y.append(label)

# converting the lists for train/test the dataset
x = np.array(x)
y = np.array(y)
# print(x)
# print(y)

x = x/255 # scaling values within 0 to 1

model = Sequential()

for i in range(2):
    model.add(Conv2D(32, (3, 3), input_shape=x.shape[1:], activation='relu')) # convolutional layer: set convolutional filter and image shape
    model.add(MaxPooling2D(pool_size=(2, 2))) # maxpooling layer: set pooling size

model.add(Flatten())

# set dense layer: output 2(e.g. Straberry_fresh is 0. 6, so that Straberry_scrotch will be 0.4)
# where its defile as softwrap as activation function
model.add(Dense(2, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=5, validation_split=0.1) # x = features(independent var) and y = label of this corresponding features(dependent var)
