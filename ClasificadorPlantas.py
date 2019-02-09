# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 13:25:19 2018

@author: Alison Ruiz
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
import cv2
import numpy as np
from keras.models import model_from_json



clasificador= Sequential()
clasificador.add(Conv2D(32,(3,3), input_shape=(64,64,3), activation='relu'))
clasificador.add(MaxPooling2D(pool_size= (2,2)))
clasificador.add(Flatten())
clasificador.add(Dense(units=150, activation='relu'))
clasificador.add(Dense(units=150, activation='relu'))
clasificador.add(Dense(units=1, activation='sigmoid')) 
clasificador.compile(optimizer='adam', loss='binary_crossentropy', metrics =['accuracy'])
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Training',
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode= 'binary')
test_set = test_datagen.flow_from_directory('Test',
                                            target_size=(64,64),
                                            batch_size = 32,
                                            class_mode ='binary')
clasificador.fit_generator(training_set,
                           steps_per_epoch = 596,
                           epochs= 13,
                           validation_data = test_set,
                           validation_steps = 180)

red1_json = clasificador.model.to_json()
with open("red_1.json", "w") as json_file:
   json_file.write(red1_json)
#serialize weights to HDF5
clasificador.model.save_weights("red_1.h5")
print("Saved red to disk")
