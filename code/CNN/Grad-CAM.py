#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2020/2/28 21:13
@Author : Jinghao Peng
@File : Grad-CAM.py
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

os.environ["CUDA_VISIBLE_DEVICES"]="7"

FLAGS = None

def Visualizing_heatmaps(model, layer_name,image_path, save_path, image_name):
    image = cv2.imread(image_path)
    img = data_process.image_normalization(image, 128, 128)
    img = np.array([img])

    # Choose the category index
    preds = model.predict(img)
    max_index = np.argmax(preds[0])
    predict_output = model.output[:, max_index]
    last_conv_layer = model.get_layer(layer_name)

    # Calculate gradients
    grads = K.gradients(predict_output, last_conv_layer.output)[0]

    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img])

    # Add gradients as weights
    conv_layer_output_value = conv_layer_output_value * pooled_grads_value

    heatmap = np.mean(conv_layer_output_value, axis=-1)


    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap)+1e-20)
    #plt.matshow(heatmap)
    #plt.axis("off")
    #plt.show()

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    overlap_img = heatmap * 0.5 + image

    # Save "heatmap" or "overlap_img"
    cv2.imwrite(os.path.join(save_path,image_name), overlap_img)
    #cv2.imwrite(os.path.join(save_path,image_name), heatmap*0.5)

def main(_):
    path = FLAGS.path
    images_dir = FLAGS.images_dir
    category = "test"

    image_lists = data_process.create_image_lists(images_dir)

    model = build_CNN()

    model.load_weights(os.path.join(path + "weights/weights", "model.h5df"))

    layer_names=["max_pool1","max_pool2","max_pool3","max_pool4"]
    #layer_names = ["max_pool4"]

    # Different conv_block with different resolutions
    for layer_name in layer_names:
        if (layer_name == "max_pool1"):
            resolution = "63"
        elif (layer_name == "max_pool2"):
            resolution = "30"
        elif (layer_name == "max_pool3"):
            resolution = "14"
        elif (layer_name == "max_pool4"):
            resolution = "6"

        for label_index, label_name in enumerate(image_lists.keys()):
            image_list = image_lists[label_name][category]
            minlist = image_list[0:120]
            for image_index, image_name in enumerate(
                    minlist):
                image_path = os.path.join(images_dir, category, label_name, image_name)

                # If you want to save "heatmap", please comment the next 2 lines.
                save_path = os.path.join(
                    path, "visualization/CAM", resolution, label_name)

                # If you want to save "heatmap", please uncomment the following.
                # save_path = os.path.join(
                #    path, "visualization/CAM/heatmap", resolution, label_name)

                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                Visualizing_heatmaps(model, layer_name, image_path, save_path, image_name)

if __name__=="__main__":
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
