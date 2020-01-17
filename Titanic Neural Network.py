# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 21:10:00 2020

@author: daniel_lee
"""

import pandas as pd
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras import losses
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.regularizers import l2, l1
from keras.callbacks import ModelCheckpoint
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

train_set = pd.read_csv('trainset.csv')
train_set.set_index(train_set.columns.values[0], drop = True, inplace = True)
X_train = train_set.drop(['Survived'], axis = 1)
Y_train = train_set['Survived']

def TitanicModel(input_row):
    X_input = Input(shape= input_row)
    X = Dense(64, activation = 'relu', use_bias = True, bias_regularizer = l2(0.01))(X_input)
    X = Dropout(.4)(X)
    X = Dense(12, activation = 'relu', use_bias = True, kernel_regularizer = l2(0.01), bias_regularizer = l2(0.01))(X)
    X = Dropout(.4)(X)
    X = Dense(6, activation = 'relu', use_bias = True, bias_regularizer = l2(0.01))(X)
    X = Dropout(.4)(X)
    X = Dense(1, activation = 'sigmoid', use_bias = True, kernel_regularizer = l2(0.01), bias_regularizer = l2(0.01))(X)
    model = Model(inputs = X_input, outputs = X, name = 'TitanicModel')
    return model

titanicmodel = TitanicModel(X_train.shape[1:])
titanicmodel.compile(loss = 'binary_crossentropy', optimizer = 'adamax', metrics = ['accuracy'])
filepath = 'weights-improvement{epoch:02d}-{val_accuracy:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor = 'val_accuracy', verbose = 1, save_best_only=True, mode = 'max')
callbacks_list = [checkpoint]

titanicmodel.fit(x = X_train, y = Y_train, epochs = 600, validation_split = 0.15, batch_size = None, callbacks = callbacks_list)
predictions = titanicmodel.predict(X_train)
comparison = pd.DataFrame({'Y_hat': predictions.ravel(), 'Y': Y_train})
comparison.to_csv('comparison.csv')
