#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2020/11/18 10:52
@Author : Jinghao Peng
@File : train.py
@Software: PyCharm
"""
import tensorflow as tf
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import h5py
import os
import argparse
import sys

import pickle

import data_process


def predict(path, images_dir, resized_width, resized_height):
    image_lists = data_process.create_image_lists(images_dir)

    test_datas, test_labels = data_process.get_batch_of_data(
        image_lists, -1, images_dir, 'test', resized_width, resized_height)

    save_model_path = os.path.join(FLAGS.path, "save_model", "SVM", "Nonlinear_SVM.pickle")

    with open(save_model_path, 'rb') as f:
        model = pickle.loads(f.read())

    pred_prob = model.predict_proba(test_datas)
    test_acc = np.mean(np.equal(np.argmax(pred_prob, axis=1), np.argmax(test_labels, axis=1)))
    print('Test accuracy:', test_acc)

    pre_data_dir = os.path.join("../../predictData/SVM")

    if not os.path.exists(pre_data_dir):
        os.makedirs(pre_data_dir)

    with h5py.File(os.path.join(pre_data_dir,"prediction_and_labels.h5"), "w") as f:
        f["prediction"] = pred_prob
        f["truth"] = test_labels


def main(argv):
    predict(FLAGS.path, FLAGS.images_dir, 64, 64)


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
