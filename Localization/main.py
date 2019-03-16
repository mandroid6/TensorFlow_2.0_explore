# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 09:44:23 2019

@author: Mandar
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Input 
from tensorflow.keras.layers import Flatten, Dropout, BatchNormalization, Concatenate, Reshape
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt 
import os 
import numpy as np
from PIL import Image
from random import shuffle
import random
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from batch_generator import batch_generator
from metric import iou
from model import Localizer

#!wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
#!wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz


batch_size = 32
IMG_SIZE = 256

all_files = os.listdir('data/annotations/xmls/')
shuffle(all_files)

split = int(0.95 * len(all_files))

train_files = all_files[0:split]
test_files  = all_files[split:]

train_generator = batch_generator(train_files, batch_size = batch_size, sz = (IMG_SIZE, IMG_SIZE ))
test_generator  = batch_generator(test_files, batch_size = batch_size, sz = (IMG_SIZE, IMG_SIZE ))


inp = Input(shape = (IMG_SIZE, IMG_SIZE, 3))
conv1 = Conv2D(16, (3, 3), padding = 'same', activation = 'relu')(inp)
pool1 = MaxPooling2D(pool_size = (2, 2))(conv1)

conv2 = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(pool1)
pool2 = MaxPooling2D(pool_size = (2, 2))(conv2)

conv3 = Conv2D(63, (3, 3), padding = 'same', activation = 'relu')(pool2)
pool3 = MaxPooling2D(pool_size = (2, 2))(conv3)

conv4 = Conv2D(128, (3, 3), padding = 'same', activation = 'relu')(pool3)
pool4 = MaxPooling2D(pool_size = (2, 2))(conv4)

conv5 = Conv2D(256, (3, 3), padding = 'same', activation = 'relu')(pool4)
pool5 = MaxPooling2D(pool_size = (2, 2))(conv5)

conv6 = Conv2D(512, (3, 3), padding = 'same', activation = 'relu')(pool5)
pool6 = MaxPooling2D(pool_size = (2, 2))(conv6)

flatten = Flatten()(pool6)
dense1 = Dense(128, activation = 'relu')(flatten)
drop1  = Dropout(0.5)(dense1)
out = Dense(4, activation = 'sigmoid')(drop1)

model = tf.keras.models.Model(inputs = inp, outputs = out)

model.compile(optimizer = 'adam' , loss = 'mean_squared_error', metrics = [iou])
model_save = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_iou', mode='max', save_weights_only= True, verbose = 0)

train_steps = len(train_files) // batch_size
test_steps = len(test_files) // batch_size