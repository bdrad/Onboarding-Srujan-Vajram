#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:27:47 2020

@author: srujanvajram
"""

# =========================================================================== # 
                    # SETTING UP IMAGE GENERATOR 
                    
import numpy as np
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator 
                    
# 1. Make sure you have organized your data into train, validation, and test sets

train_directory = ""
validation_directory = ""
test_directory = ""

# We now create the BATCHES of data using Image Generator

# For the TRAIN BATCH 
train_batch = ImageDataGenerator(
    preprocessing_function = None).flow_from_directory(
        directory = train_directory,
        target_size = (224,224),
        classes = ['class_A', 'class_B'],
        batch_size = 10,
        shuffle=True)
        
validation_batch = ImageDataGenerator(
    preprocessing_function = None).flow_from_directory(
        directory = validation_directory,
        target_size = (224,224),
        classes = ['class_A', 'class_B'],
        batch_size = 10,
        shuffle=True)
        
test_batch = ImageDataGenerator(
    preprocessing_function = None).flow_from_directory(
        directory = test_directory,
        target_size = (224,224),
        classes = ['class_A', 'class_B'],
        batch_size = 10,
        shuffle=False)
    
# The only difference for the test batch, is we specify shuffle = False so that we can 
# later create a confusion matrix (for which we need unshuffled data.) 

# =========================================================================== # 
                                # SETTING UP CNN

# Few things to note. kernel_size refers to the filter size. Padding = 'same' 
# implies that zero padding will be added to the image so that its dimensions 
# are preserved 
                                
model = Sequential([
    Conv2D(  filters=32, kernel_size = (3,3), activation = 'relu', padding = 'same', input_shape = (224,224,3)  ),
    MaxPool2D(  pool_size = (2,2), strides = 2  ),
    Flatten(),
    Dense(units = 2, activation = 'softmax'),
    ])

# We compile our model 
model.compile(optimizer = Adam(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
                
# Important note about losses: 
# For 2 CLASSES, you can use BINARY CROSSENTROPY for the loss. You would only have ONE 
# unit in the output layer, and your activation function would be SIGMOID 

# For MORE THAN 2 CLASSES, you can use CATEGORICAL CROSSENTROPY, just make sure the last
# layer unit number is the same as the number of classes, and use softmax. You can 
# STILL USE CATEGORICAL for ONLY 2 CLASSES. YOu just have to set it up like above
# CATEGORICAL CROSSENTROPY is generally used for multiclass classification. 

# Finally, fit the model based on the generators you created. 
# We pass training data to x, validation data to validation_data, and finally
# Specify epochs and verbose 
model.fit(x = train_batch, validation_data = validation_batch, epochs = 2, verbose = 2)
    