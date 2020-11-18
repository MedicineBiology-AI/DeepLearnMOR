#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2020/11/18 10:52
@Author : Jinghao Peng
@File : train.py
@Software: PyCharm
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import h5py

from tensorflow.contrib.tensor_forest.python import tensor_forest

def randomforest(x,y,features_dim,class_num,tree_num):
    with tf.name_scope('random_forest'):
        Hparams = tensor_forest.ForestHParams(num_classes=class_num,
                                              num_features=features_dim,
                                              num_trees=tree_num).fill()

        forest_graph = tensor_forest.RandomForestGraphs(Hparams)

    train_step = forest_graph.training_graph(x, y)

    with tf.name_scope('random_forest_loss'):
        loss = forest_graph.training_loss(x, y)
        tf.summary.scalar("svm_loss", loss)

    with tf.name_scope('accuracy'):
        output, _, _ = forest_graph.inference_graph(x)
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.cast(y, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    return train_step,loss,accuracy,output