#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2020/2/28 21:13
@Author : Jinghao Peng
@File : hidden_layer_output_visualization.py
@Software: PyCharm
"""
import tensorflow as tf
import numpy as np
import os
import h5py
import cv2
import argparse
import sys
import matplotlib.pyplot as plt

import data_process
from build_model import build_CNN
from keras.models import Sequential, Model

os.environ["CUDA_VISIBLE_DEVICES"]="0"

tf.reset_default_graph()

FLAGS = None

def conv_output(model, layer_name, img):
    input_img = model.input

    try:
        out_conv = model.get_layer(layer_name).output
    except:
        raise Exception("No layer named {}!".format(layer_name))

    intermediate_layer_model = Model(inputs=input_img, outputs=out_conv)

    intermediate_output = intermediate_layer_model.predict(img)

    return intermediate_output

def image_show(feature_map,channel_num,save_path,image_name):
    # Rows represents the number of subfigures in each row
    rows = 16
    cols = int(channel_num / rows)

    feature_map_combination = []
    # You can choose the shape of the figure
    fig = plt.figure(figsize=(10,10))
    for i in range(channel_num):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(feature_map_split)
        plt.axis("off")

    plt.savefig(os.path.join(save_path,image_name))
    #plt.show()
    plt.close()

def activation_max_show(feature_map,save_path,image_name):
    activations=np.sum(feature_map,axis=(0,1))
    index=np.argsort(activations)

    feature_map_max=feature_map[:,:,index[-1]]

    plt.figure(figsize=(4,4))
    plt.imshow(feature_map_max)
    plt.axis("off")

    plt.savefig(os.path.join(save_path,image_name))
    #plt.show()
    plt.close()

def main(_):
    path = FLAGS.path
    images_dir = FLAGS.images_dir
    category = "test"

    image_lists = data_process.create_image_lists(images_dir)

    model = build_CNN()

    model.load_weights(os.path.join(path,"weights/weights", "model.h5df"))

    # Select a different intermediate layer and change the output figure size in "image_show".
    #layer_names=["max_pool1","max_pool2","max_pool3","max_pool4"]
    layer_names = ["max_pool4"]

    image_num = 120

    images = []
    file_names = []
    for label_index, label_name in enumerate(image_lists.keys()):
        image_list = image_lists[label_name][category]
        minlist = image_list[0:image_num]
        for image_index, image_name in enumerate(
                minlist):
            image_path = os.path.join(images_dir, category, label_name, image_name)
            image_data = cv2.imread(image_path)
            images.append(data_process.image_normalization(
                image_data, 128, 128))
            file_names.append(image_name)
    images = np.array(images)
    file_names = np.array(file_names)

    print(images.shape)
    print(file_names.shape)

    for layer_name in layer_names:
        for label_index, label_name in enumerate(image_lists.keys()):
            img_inputs = images[image_num * label_index:image_num * (label_index + 1)]
            img_names = file_names[image_num * label_index:image_num * (label_index + 1)]
            feature_map = conv_output(model, layer_name, img_inputs)

            # If you want to save the channel with maximum activation, please comment the next 2 lines.
            save_path = os.path.join(
                path,"visualization/Hidden_layer", layer_name, label_name)

            # If you want to save the channel with maximum activation, please uncomment the next 2 lines.
            # save_path = os.path.join(
            #    path, "visualization/Hidden_layer/activation_max", layer_name, label_name)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            for i in range(image_num):
                image = feature_map[i]
                image_name = img_names[i]
                # If you want to save the channel with maximum activation, please comment the next lines.
                image_show(image, image.shape[-1], save_path, image_name)

                # If you want to save the channel with maximum activation, please uncomment the next lines.
                # activation_max_show(image, save_path, image_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="./",
        help="The absolute path of the CNN."
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="../../DeepOrganelleDataset",
        help="Images folder directory."
    )
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
