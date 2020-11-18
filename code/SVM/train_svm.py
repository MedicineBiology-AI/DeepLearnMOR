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
from sklearn import svm

import pickle

import data_process

FLAGS = None


def main(_):
    resized_width = 64
    resized_height = 64

    since = time.time()

    image_lists = data_process.create_image_lists(FLAGS.images_dir)

    test_datas, test_labels = data_process.get_batch_of_data(
        image_lists, -1, FLAGS.images_dir, 'test', resized_width, resized_height)

    train_datas, train_labels = data_process.get_batch_of_data(
        image_lists, -1, FLAGS.images_dir, 'train', resized_width, resized_height)

    print(train_datas.shape)
    print(train_labels.shape)

    model = svm.SVC(kernel='rbf', verbose=True, probability=True)
    model.fit(train_datas, np.argmax(train_labels, axis=1))

    pred_prob = model.predict_proba(test_datas)
    test_acc = np.mean(np.equal(np.argmax(pred_prob, axis=1), np.argmax(test_labels, axis=1)))
    print('Test accuracy:', test_acc)

    save_model_path = os.path.join(FLAGS.path, "save_model", "SVM", "Nonlinear_SVM.pickle")

    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    with open(save_model_path, 'wb') as f:
        pickle.dump(model, f)

    time_elapsed = time.time() - since

    print("Runtime: {}min, {:0.2f}sec".format(int(time_elapsed // 60), time_elapsed % 60))

    save_results_path = os.path.join(FLAGS.path, "results/SVM", "results.txt")

    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)

    with open(save_results_path, "w") as f:
        f.write("Test accuracy =  {}".format(test_acc) + "\n")
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
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
