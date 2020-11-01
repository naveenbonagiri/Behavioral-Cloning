import csv
import cv2
import numpy as np
import sklearn
import math
import gc

import tensorflow as tf
import pandas as pd
import json
import random
import csv

from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D, Cropping2D
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

def process_csvfile():
    all_data = []

    gc.collect()
    
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for data in reader:
            all_data.append(data)

    # exclude header row from the csv
    all_data = all_data[1:]

    return all_data

def split_data(alldata):
    gc.collect()
    
    # split data into training and validation sets 80%, 20% respectively
    training_data, validation_data = train_test_split(alldata, test_size=0.2)
        
    return training_data, validation_data 

def process_data_by_data_by_batch(batch):
    
    gc.collect()
    
    images = []
    steering_angledata = []
    
    # loop through each row of the batch_data and create array of 
    # images and steering angle data. process center, left, right
    # portions camera data for the batch.
    for data in batch:
        path = 'data/IMG/'
        center_img = cv2.cvtColor(cv2.imread(path + data[0].split('/')[-1]), cv2.COLOR_BGR2RGB)
        left_img = cv2.cvtColor(cv2.imread(path + data[1].split('/')[-1]), cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(cv2.imread(path + data[2].split('/')[-1]), cv2.COLOR_BGR2RGB)

        #steering angle processing
        steering_angle_center = float(data[3])

        # create adjusted steering measurements for the side camera images
        correction = 0.2 # this is a parameter to tune
        steering_angle_left = steering_angle_center + correction
        steering_angle_right = steering_angle_center - correction

        images.extend([center_img, left_img, right_img])
        steering_angledata.extend([steering_angle_center, steering_angle_left, steering_angle_right])

    augmented_images, augmented_steering_angledata = [], []
    for image,steering_angle in zip(images, steering_angledata):
        augmented_images.append(cv2.flip(image,1))
        augmented_steering_angledata.append(steering_angle*-1.0)

    return augmented_images, augmented_steering_angledata, images, steering_angledata

def data_generator(data_set, batch_size = 16):
    total_images = []
    total_angles = []

    gc.collect()
    
    data_set_length = len(data_set)

    # never terminate and generate data for each call
    # shuffle the data and process batch data based on the batch size
    while 1: 
         data_set = sklearn.utils.shuffle(data_set)        
                
         for offset in range(0, data_set_length, batch_size):
            batch_data = data_set[offset : offset + batch_size]	           
          
            augmented_images, augmented_steering_angle, images, steering_angledata = process_data_by_data_by_batch(batch_data)
            
            total_images.extend(images)  
            total_images.extend(augmented_images) 
            total_angles.extend(steering_angledata)  
            total_angles.extend(augmented_steering_angle)   
            
            X_train_batch, Y_train_batch = np.asarray(total_images), np.asarray(total_angles)

            yield sklearn.utils.shuffle(X_train_batch, Y_train_batch) 

def run_cnn():
    gc.collect()
    model = Sequential()
    #model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    #remove the sky and other unwanted parts of the image
    model.add(Cropping2D(cropping=((70, 25),(0,0))))

    #model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"), W_regularizer=l2(0.001))
    #model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"), W_regularizer=l2(0.001))
    #model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"), W_regularizer=l2(0.001))

    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))

    #model.add(Convolution2D(64,3,3,activation="relu"), W_regularizer=l2(0.001))
    #model.add(Convolution2D(64,3,3,activation="relu"), W_regularizer=l2(0.001))

    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))

    model.add(Flatten())
    #model.add(Dense(100), W_regularizer=l2(0.001))
    model.add(Dense(100))
    model.add(Dropout(0.20))
    #model.add(Dense(50), W_regularizer=l2(0.001))
    model.add(Dense(50))
    model.add(Dropout(0.20))
    #model.add(Dense(10), W_regularizer=l2(0.001))
    model.add(Dense(10))
    model.add(Dropout(0.20))
    model.add(Dense(1))

    return model

gc.collect()

#process csv file
alldata = process_csvfile()

gc.collect()

#split the data into training, validation sets
trainingdata, validationdata = split_data(alldata)

print("Training Data Size:", len(trainingdata))
print("Validation Data Size:", len(validationdata))

gc.collect()

train_generator_data = data_generator(trainingdata, batch_size=16)
validation_generator_data = data_generator(validationdata, batch_size=16)

gc.collect()

model = run_cnn()

#model.compile(loss='mse', optimizer ='adam')
model.compile(loss='mse', optimizer = Adam(lr=1e-4))
model.summary()

gc.collect()

batch_size = 16

#history = model.fit_generator(train_generator_data, steps_per_epoch=math.ceil(len(trainingdata)/batch_size), validation_data=validation_generator_data, validation_steps=math.ceil(len(validationdata)/batch_size), epochs = 3, verbose = 1)
 
history = model.fit_generator(train_generator_data, steps_per_epoch=16, validation_data=validation_generator_data, validation_steps=16, epochs = 3, verbose = 1)

model.save('model.h5')

# print keys from history object
print(history.history.keys())

# plot each epoch training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss.png')