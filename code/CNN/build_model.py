#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2020/2/28 21:13
@Author : Jinghao Peng
@File : build_model.py
@Software: PyCharm
"""
import keras
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,  Flatten
from keras.optimizers import Adam
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
def build_CNN():
    img = Input(shape=(128, 128, 3))
    X = Conv2D(filters=32, kernel_size=[3, 3], padding="valid",name="cnn1")(img)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="valid",name="max_pool1")(X)

    X = Conv2D(filters=64, kernel_size=[3, 3], padding="valid",name="cnn2")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="valid",name="max_pool2")(X)

    X = Conv2D(filters=128, kernel_size=[3, 3], padding="valid",name="cnn3")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="valid",name="max_pool3")(X)
    
    X = Conv2D(filters=256, kernel_size=[3, 3], padding="valid",name="cnn4")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="valid",name="max_pool4")(X)

    X = Dropout(0.5)(X)

    X = Flatten()(X)

    X = Dense(units=512,name="fc1")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Dropout(0.5)(X)
    
    X = Dense(units=256,name="fc2")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Dropout(0.5)(X)

    classifier = Dense(units=6,activation="softmax",name="classifier")(X)

    model = Model(img, classifier)

    return model

if __name__=="__main__":
    model=build_CNN()
    model.summary()