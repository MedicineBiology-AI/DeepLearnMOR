#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2020/2/28 21:13
@Author : Jinghao Peng
@File : predict.py
@Software: PyCharm
"""
import tensorflow as tf
import numpy as np
import h5py
import os
import argparse
import sys

import data_process
from build_model import build_CNN
from keras.optimizers import Adam

os.environ["CUDA_VISIBLE_DEVICES"]="0"

FLAGS = None

def main(_):
    resized_width = 128
    resized_height = 128

    model = build_CNN()

    opt = Adam(lr=FLAGS.learning_rate)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])

    model.load_weights(os.path.join(FLAGS.path,"weights/weights/", "model.h5df"))

    image_lists = data_process.create_image_lists(FLAGS.images_dir)

    test_datas, test_labels = data_process.get_batch_of_data(
        image_lists, -1, FLAGS.images_dir, "test", resized_width, resized_height)

    test_loss, test_acc = model.evaluate(test_datas, test_labels)

    print("Test accuracy:{0:.4f}, test loss:{1:.4f}".format(test_acc, test_loss))
    
    prediction=model.predict(test_datas, batch_size=FLAGS.batch_size)

    pre_data_dir = "../../predictData/CNN"

    if not os.path.exists(pre_data_dir):
        os.makedirs(pre_data_dir)

    with h5py.File(os.path.join(pre_data_dir,"prediction_and_labels.h5"),"w") as f:
        f["prediction"]=prediction
        f["truth"]=test_labels

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
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="The number of epochs. Each epoch contains 100 iterations"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="How many images to test on at a time."
    )
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)