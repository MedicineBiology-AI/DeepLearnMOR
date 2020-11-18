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

FLAGS = None


def main(_):
    resized_width = 128
    resized_height = 128

    since = time.time()

    X = tf.placeholder(tf.float32, shape=[None, 128 * 128 * 3])
    Y = tf.placeholder(tf.float32, shape=[None])

    image_lists = data_process.create_image_lists(FLAGS.images_dir)

    class_count = len(image_lists.keys())

    train_step, randomforest_loss, accuracy = randomforest(X, Y, 128 * 128 * 3, class_count, FLAGS.tree_num)

    saver = tf.train.Saver()

    save_model_path = os.path.join(FLAGS.path, "save_model", "RandomForest")
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    save_results_path = os.path.join(FLAGS.path, "results", "results.txt")
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)

    init = tf.group(tf.global_variables_initializer(), resources.initialize_resources(resources.shared_resources()))

    with tf.Session(config=config) as sess:
        sess.run(init)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.path + "results/logs/train", sess.graph)

        validation_writer = tf.summary.FileWriter(FLAGS.path + "results/logs/validation")

        val_datas, val_labels = data_process.get_batch_of_data(
            image_lists, -1, FLAGS.images_dir, "val", resized_width, resized_height)

        val_labels = np.argmax(val_labels, axis=1)

        test_datas, test_labels = data_process.get_batch_of_data(
            image_lists, -1, FLAGS.images_dir, "test", resized_width, resized_height)

        test_labels = np.argmax(test_labels, axis=1)

        best_acc = 0.0

        for i in range(FLAGS.iters):
            train_datas, train_labels = data_process.get_batch_of_data(
                image_lists, FLAGS.batch_size, FLAGS.images_dir, "train", resized_width, resized_height)

            train_labels = np.argmax(train_labels, axis=1)

            sess.run(train_step,
                     feed_dict={X: train_datas, Y: train_labels})

            if (i % 10 == 0):
                train_loss, train_acc = sess.run([randomforest_loss, accuracy],
                                                 feed_dict={X: train_datas, Y: train_labels})
                train_summary = sess.run(merged, feed_dict={X: train_datas, Y: train_labels})

                val_loss, val_acc = sess.run([randomforest_loss, accuracy],
                                             feed_dict={X: val_datas, Y: val_labels})
                val_summary = sess.run(merged, feed_dict={X: val_datas, Y: val_labels})

                train_writer.add_summary(train_summary, i)
                validation_writer.add_summary(val_summary, i)

                if (val_acc > best_acc):
                    best_acc = val_acc
                    test_loss1, test_acc1 = sess.run(
                        [randomforest_loss, accuracy],
                        feed_dict={X: test_datas, Y: test_labels})
                    saver = tf.train.Saver()
                    saver.save(sess, save_model_path)

                print("Iteration {0:d}: train loss:{1:f}, train acc:{2:f}, val loss:{3:f}, val acc:{4:f}".format(
                    i, train_loss, train_acc, val_loss, val_acc))

        test_loss2, test_acc2 = sess.run([randomforest_loss, accuracy],
                                         feed_dict={X: test_datas, Y: test_labels})

        print("Best validation accuracy = {}".format(best_acc * 100))
        print("Test accuracy with best validation =  {}".format(test_acc1 * 100))
        print("Final test accuracy =  {}".format(test_acc2 * 100))

    time_elapsed = time.time() - since

    print("Runtime: {}min, {:0.2f}sec".format(int(time_elapsed // 60), time_elapsed % 60))

    with open(save_results_path, "w") as f:
        f.write("Best validation accuracy: " + str(best_acc) + "\n")
        f.write("Test accuracy with best validation: " + str(test_acc1) + "\n")
        f.write("Final test accuracy =  {}".format(test_acc2 * 100) + "\n")
        f.write("Runtime: " + str(int(time_elapsed // 60)) + "min," + str(time_elapsed % 60) + "sec")


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
    parser.add_argument(
        "--iters",
        type=int,
        default=5000,
        help="The number of epochs. Each epoch contains 100 iterations"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="How many images to train on at a time."
    )
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
