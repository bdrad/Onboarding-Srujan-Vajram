#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 22:35:32 2020

@author: srujanvajram
"""

# =========================================================================== # 
#                       TENSORFLOW DATA BASICS 

import numpy as np 
from sklearn.utils import shuffle 
from sklearn.preprocessing import MinMaxScaler

# Tensorflow works with numpy arrays. As such, if you have any normal lists,
# you should first convert them into numpy arrays. Ex. 

# Numbers 
train_samples = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

# Label as either odd (0) or even (1)
train_labels = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]

# CONVERT to numpy arrays
train_samples = np.array(train_samples)
train_labels = np.array(train_labels) 

# It is always good practice to SHUFFLE your data
# This will shuffle the label data and sample data the SAME way (order preserved)
train_labels, train_samples = shuffle(train_labels,train_samples)

# It is beneficial to scale our data (normalize.) We can use the MinMaxScaler
# feature from sklearn to scale the data between two values as such

# Specify the range that you want your data to subsume 
scaler = MinMaxScaler(feature_range=(0,1)) 

# Scale the samples. Note there are a  few things happening here.

# 1. We are invoking 'fit_transform' which is the main method that is 'transforming' the 
# data to the desired range
# 2. We are reshaping train_samples using reshape(-1,1)
# the '-1' simply tells numpy to figure out what dimension is needed to reshape 
# the array to something valid based on the other dimension. In this case,
# we are reshaping train_samples to a column array. 

scaled_training_samples = scaler.fit_transform(train_samples.reshape(-1,1))

# =========================================================================== # 
#                   CREATING A SIMPLE SEQUENTIAL MODEL

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Activation, Dense 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy 


# We are creating a sequential model (layer by layer containing hidden units)
# Notice the first layer always specifies the input hsape 
# Understand the DIFFERENCES between the diffenret types of losses and when to use them.
# Some great resources 
# https://gombru.github.io/2018/05/23/cross_entropy_loss/
# https://stats.stackexchange.com/questions/260505/should-i-use-a-categorical-cross-entropy-or-binary-cross-entropy-loss-for-binary   
    
model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'), 
    Dense(units=32, activation='relu'), 
    Dense(units=2, activation='softmax')
    ])
    
# Will summarize your model
model.summary()

# Now we want to COMPILE our model to set up the tensor pipelines. 
# sparse categorical is used when you are not using a one-hot encoding for labels (but integers instead)
# Ex, windy,cloudy,sun. one hot for cloudy=[0,1,0]. Were using '2' for cloudy rn.
# If you are using one hot enconding, use categorical crossentropy 
model.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Then we FIT the data to the model to 'train' the weights 
# verbose will allow us to see output messages 
model.fit(x = scaled_training_samples, y = train_labels, batch_size=10, epochs=10000,shuffle=True, verbose=2)
    
# =========================================================================== # 
#                        VALIDATION DATA SETS FROM TRAINING DATA

# You can create a validation set from the training set when you invoke model.fit
# To do this, you use validation_split, which will take the last X% of the training data as the validation data
# An important point to keep in mind. The data will NOT be shuffled prior to invoking validation_split (even when shuffle=true)
# Make sure the training data has a good distribution before invoking it. 
model.fit(x = scaled_training_samples, y = train_labels, validation_split = 0.1, batch_size=10, epochs=10000,shuffle=True, verbose=2)

# =========================================================================== # 
#                       USING  MODEL TO PREDICT ON TEST DATA

# To make predictions using our model, we invoke the model.predict function

predictions = model.predict(x = scaled_test_samples, batch_size = 10, verbose = 0)

# Your predictions will likely be in the form of probabilities (for each class)
# If you want to pick the largest probability from each prediction
# you can use the np.argmax function
# Note that axis=-1 just tells argmax to look at the elemenets in the last dimension (in this case, columns)
predictions = np.argmax(predictions, axis=-1)

# =========================================================================== # 
#                        HOW TO SAVE AND LOAD A MODEL

import os.path

# Set the path of where you want to save the model
path = '/Users/srujanvajram/Documents/Internship related/'

# Name your model. Note the 'h5' extension which is for Hierarchical Data Format 
# This format is used to store multidimensional array data. 
model_name = 'hand_recognition.h5'

# Concatenate 
model_path = path + model_name
        
# We invoke model.save() to save the model. Note this will save
# the most recently set up model in the code. 
# This function saves 4 things
# 1. The architecture of the model (layers, activations)
# 2. The weights
# 3. Training config (optimizer, loss...)
# 4. Training state. i.e you can contine training where you left off. 
model.save(model_path)

# We import the load_model function
from tensorflow.keras.models import load_model

# We load the model using its path 
new_model = load_model(model_path)

# We can inspect thw weights onf the model
new_model.get_weights()

# We can inspect the optimizer
new_model.optimizer

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  # 

# If you are only concerned with model architecture (and not weights)
# you can invoke the to_json function. 
# This function will summarize the model parameters in a string.
# This string can later be used to reoconstruct the architecture
json_string = model.to_json

# We can use the string to reinitialize a model with the same archictecture
from tensorflow.keras.models import model_from_json
model_architecture = model_from_json(json_string)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  # 

# If we only want to save the weights, we can do that too
model.save_weights(model_path)

# NOTE: prior to loading the weights, you need to RE-SETUP the model architecture
# as done in line 68. We did not save the model architecture
# AFTER you set up the model again, you can invoke load_weights
model_2 = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'), 
    Dense(units=32, activation='relu'), 
    Dense(units=2, activation='softmax')
    ])

# NOTICE that the architecture needs to match the saved model in order
# to load the weights 
model_2.load_weights(model_path)

# =========================================================================== # 































