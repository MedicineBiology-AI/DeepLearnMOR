#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2020/2/28 21:13
@Author : Jinghao Peng
@File : ROC.py
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
from sklearn.metrics import roc_curve, auc

FLAGS = None

def draw_roc_for_class(prediction, truth, class_list):
    plt.figure(figsize=(10, 8))
    for index, class_name in enumerate(class_list):
        FPR, TPR, thresholds = roc_curve(truth[:, index], prediction[:, index])
        roc_auc = auc(FPR, TPR)
        plt.plot(FPR, TPR, label="{0:s} (AUC = {1:.4f})".format(class_name, roc_auc))

    plt.plot([0, 1], [0, 1], ":", color=(0.6, 0.6, 0.6), label="reference")
    plt.xlim([-0.02, 1.05])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(" ROC curves")
    plt.legend(loc="lower right",fontsize=11.5)
    plt.show()

def main(_):
    path=FLAGS.path
    model_name = FLAGS.model_name

    with h5py.File(os.path.join(path, "predictData", model_name, "prediction_and_labels.h5"), "r") as f:
        prediction = np.array(f["prediction"])
        truth = np.array(f["truth"])

    with open(os.path.join(path,model_name, "output_labels.txt"), "r") as f:
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

        # Draw ROC curve for each class of a model
        draw_roc_for_class(prediction, truth, class_list)

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