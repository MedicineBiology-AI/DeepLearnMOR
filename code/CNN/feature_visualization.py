#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2020/2/28 21:13
@Author : Jinghao Peng
@File : feature_visualization.py
@Software: PyCharm
"""
import tensorflow as tf
import numpy as np
import keras
import os
import cv2
import argparse
import sys

import data_process
from build_model import build_CNN
from keras import backend as K
import matplotlib.pyplot as plt
import time

os.environ["CUDA_VISIBLE_DEVICES"]="0"

FLAGS = None

def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

def deprocess_image(img):
    img = img - img.mean()
    img = img / (img.std() + 1e-5)
    img = img * 0.1 + 0.5
    img = np.clip(img, 0, 1)

    img = img * 255
    if K.image_data_format() == "channels_first":
        img = img.transpose((1, 2, 0))
    img = np.clip(img, 0, 255).astype("uint8")

    return img

def conv_filter(model, layer_name, img):
    input_img = model.input

    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

    try:
        layer_output = layer_dict[layer_name].output
    except:
        raise Exception("Not layer named {}!".format(layer_name))

    sort_filters = []

    for i in range(layer_output.shape[-1]):
        loss = K.mean(layer_output[:, :, :, i])
        # Calculate gradients and normalize it
        grads = K.gradients(loss, input_img)[0]
        grads = normalize(grads)

        iterate = K.function([input_img], [loss, grads])
        # Gradient ascent
        step = 1.
        filter_image = img.copy()
        for j in range(64):
            loss_value, grads_value = iterate([filter_image])
            filter_image += grads_value * step

        filter_image = deprocess_image(filter_image[0])
        sort_filters.append((filter_image, loss_value))
        #sort_filters.append(filter_image)

        # Sorting activation values
        sort_filters.sort(key=lambda x: x[1], reverse=True)

    return np.array([f[0] for f in sort_filters])

def filters_show(filters,filter_num,save_path,image_name):
    rows = 4
    cols = int(filter_num / rows)

    filter_combination = []
    fig = plt.figure(figsize=(10,10))

    for i in range(filter_num):
        filter_split = filters[i]
        filter_combination.append(filter_split)
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(filter_split)
        plt.axis("off")

    plt.savefig(os.path.join(save_path,image_name))
    #plt.show()
    plt.close()

def main(_):
    path = FLAGS.path
    image_name="random"

    # Generate a random image
    img = np.random.random((128, 128, 3))
    # cv2.imwrite(os.path.join(path,"visualization/filter_sample","random_image.jpg"),img*255)
    img = np.array([img])

    model = build_CNN()

    model.load_weights(os.path.join(path, "weights/weights", "model.h5df"))

    # Select the corresponding conv_block
    layer_names = ["max_pool1", "max_pool2", "max_pool3", "max_pool4"]
    #layer_names = ["max_pool1"]

    for layer_name in layer_names:
        save_path = os.path.join(
            path + "visualization/filter_sample", image_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        filters = conv_filter(model, layer_name, img)

        filters_show(filters, FLAGS.filter_num, save_path, layer_name + ".jpg")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="./",
        help="The absolute path of the CNN."
    )
    parser.add_argument(
        "--filter_num",
        type=int,
        default=16,
        help="The absolute path of the CNN."
    )
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)