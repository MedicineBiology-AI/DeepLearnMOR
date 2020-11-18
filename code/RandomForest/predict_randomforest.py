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
import time
import argparse
import sys

from tensorflow.python.ops import resources
from tensorflow.contrib.tensor_forest.python import tensor_forest

import data_process
from build_randomforest import randomforest

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.reset_default_graph()


def predict(path, images_dir, tree_num, resized_width, resized_height):
    X = tf.placeholder(tf.float32, shape=[None, 128 * 128 * 3])
    Y = tf.placeholder(tf.float32, shape=[None])

    image_lists = data_process.create_image_lists(images_dir)

    train_step, randomforest_loss, accuracy, output = randomforest(X, Y, 128 * 128 * 3, 6, tree_num)

    saver = tf.train.Saver()

    init = tf.group(tf.global_variables_initializer(), resources.initialize_resources(resources.shared_resources()))

    with tf.Session(config=config) as sess:
        sess.run(init)

        ckpt = tf.train.latest_checkpoint(os.path.join(path, 'save_model'))
        print(ckpt)
        saver.restore(sess, ckpt)

        test_datas, test_labels = data_process.get_batch_of_data(
            image_lists, -1, images_dir, 'test', resized_width, resized_height)

        labels = np.argmax(test_labels, axis=1)

        print(test_datas.shape)
        print(test_labels.shape)
        print(labels.shape)

        test_acc, prediction = sess.run([accuracy, output], feed_dict={X: test_datas, Y: labels})
        print('Test accuracy:{0:.4f}'.format(test_acc))

        pre_data_dir = os.path.join("../../predictData/RandomForest")

        if not os.path.exists(pre_data_dir):
            os.makedirs(pre_data_dir)

        with h5py.File(os.path.join(pre_data_dir,"prediction_and_labels.h5"), "w") as f:
            f["prediction"] = prediction
            f["truth"] = test_labels


def main(argv):
    predict(FLAGS.path, FLAGS.images_dir, FLAGS.tree_num, 128, 128)


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
    parser.add_argument(
        "--tree_num",
        type=int,
        default=64,
        help="The number of decision trees in random forest."
    )
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
