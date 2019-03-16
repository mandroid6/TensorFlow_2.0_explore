# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 10:14:12 2019

@author: Mandar
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Input 
from tensorflow.keras.layers import Flatten, Dropout, BatchNormalization, Concatenate, Reshape

# subclassing style for creating models
class Localizer(tf.keras.Model):
    def __init__(self, input_shape):
        super(Localizer, self).__init__()
        
        #self.input = Input(shape=input_shape)
        self.conv1 = Conv2D(16, (3, 3), padding = 'same', activation = 'relu')
        self.pool1 = MaxPooling2D(pool_size = (2, 2))
    
        self.conv2 = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')
        self.pool2 = MaxPooling2D(pool_size = (2, 2))
    
        self.conv3 = Conv2D(63, (3, 3), padding = 'same', activation = 'relu')
        self.pool3 = MaxPooling2D(pool_size = (2, 2))
    
        self.conv4 = Conv2D(128, (3, 3), padding = 'same', activation = 'relu')
        self.pool4 = MaxPooling2D(pool_size = (2, 2))
    
        self.conv5 = Conv2D(256, (3, 3), padding = 'same', activation = 'relu')
        self.pool5 = MaxPooling2D(pool_size = (2, 2))
    
        self.conv6 = Conv2D(512, (3, 3), padding = 'same', activation = 'relu')
        self.pool6 = MaxPooling2D(pool_size = (2, 2))
        
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation = 'relu')
        self.drop1  = Dropout(0.5)
        
        self.out = Dense(4, activation = 'sigmoid')
     
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.pool4(x)
        
        x = self.conv5(x)
        x = self.pool5(x)
        
        x = self.conv6(x)
        x = self.pool6(x)
        
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.drop1(x)
        
        out = self.out(x)
        
        return out