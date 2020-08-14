#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2020/2/28 21:13
@Author : Jinghao Peng
@File : confusion_matrix.py
@Software: PyCharm
"""
import tensorflow as tf
import seaborn as sns
import numpy as np
import h5py
import argparse
import sys
import os
import matplotlib.pyplot as plt

def confusion_matrix(prediction, truth, class_list):
    class_count = len(class_list)
    CM = np.zeros([class_count, class_count])
    for i in range(truth.shape[0]):
        CM[truth[i]][prediction[i]] += 1

    CM = np.array(CM, dtype=np.int32)

    f, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(CM, annot=True, center=70, fmt="d", linecolor="white", cmap="Purples", ax=ax)
    ax.set_xticklabels(class_list, rotation=35, fontsize=11)
    ax.set_yticklabels(class_list, rotation=0, fontsize=11)
    ax.set_xlabel("Prediction Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=12)
    plt.show()

    # return CM

def main(_):
    path = FLAGS.path
    model_name = FLAGS.model_name

    with h5py.File(os.path.join(path, "predictData", model_name, "prediction_and_labels.h5"), "r") as f:
        prediction = np.array(f["prediction"])
        truth = np.array(f["truth"])

    with open(os.path.join(path, "output_labels.txt"), "r") as f:
        class_list = [line.strip("\n") for line in f.readlines()]

    y_pred = tf.placeholder(tf.float32, [None, len(class_list)])
    y_truth = tf.placeholder(tf.float32, [None, len(class_list)])

    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_truth, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_truth, logits=y_pred))

    with tf.Session() as sess:
        pre_labels, truth_labels = sess.run([tf.argmax(y_pred, 1), tf.argmax(y_truth, 1)],
                                             feed_dict={y_pred: prediction, y_truth: truth})
        accuracy, loss = sess.run([acc, cross_entropy], feed_dict={y_pred: prediction, y_truth: truth})
        print("accuracy: {0:.4f}, entropy_cross: {1:.4f}".format(accuracy, loss))

        confusion_matrix(pre_labels, truth_labels, class_list)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="../",
        help="The absolute path of your project."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="densenet169",
        help="The name of the model."
    )
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)