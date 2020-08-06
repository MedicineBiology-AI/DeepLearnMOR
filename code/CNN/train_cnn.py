#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2020/2/28 21:13
@Author : Jinghao Peng
@File : train.py
@Software: PyCharm
"""
import tensorflow as tf
import numpy as np
import h5py
import os
import time
import argparse
import sys

from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard

import data_process
from build_model import build_CNN

from tensorflow.python.platform import gfile

os.environ["CUDA_VISIBLE_DEVICES"]="0"

FLAGS = None

def generate_train_data(image_lists,images_dir,batch_size,resized_width, resized_height):
    while(True):
        train_datas, train_labels = data_process.get_batch_of_data(
            image_lists, batch_size, images_dir, "train", resized_width, resized_height)

        yield (train_datas, train_labels)

class get_result(Callback):
    def on_epoch_end(self,epoch,logs={}):
        #print("train_loss:",logs.get("loss"))
        #print("train_acc:",logs.get("acc"))
        print("val_loss:",logs.get("val_loss"))
        print("val_acc",logs.get("val_acc"),"\n")

def main(_):
    resized_width = 128
    resized_height = 128

    since = time.time()

    model = build_CNN()
    model.summary()

    opt = Adam(lr=FLAGS.learning_rate)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])

    result=get_result()

    checkpoint_path=os.path.join(FLAGS.path,"save_model", "CNN.h5df")
    checkpoint=ModelCheckpoint(filepath=checkpoint_path,
                               save_best_only=True,
                               save_weights_only=True,
                               monitor="val_acc",
                               mode=max)

    tb=TensorBoard(log_dir=os.path.join(FLAGS.path,"results/results/logs"))

    callbacks=[result,checkpoint,tb]

    image_lists=data_process.create_image_lists(FLAGS.images_dir)

    with gfile.FastGFile(os.path.join(FLAGS.path,"results/output_labels.txt"), "w") as f:
          f.write("\n".join(image_lists.keys()) + "\n")

    val_datas, val_labels = data_process.get_batch_of_data(
            image_lists, -1, FLAGS.images_dir, "val", resized_width, resized_height)

    model.fit_generator(
        generate_train_data(image_lists,FLAGS.images_dir,FLAGS.batch_size,resized_width, resized_height),
        epochs=FLAGS.epochs,
        steps_per_epoch=100,
        validation_data=(val_datas, val_labels),
        callbacks=callbacks
    )

    test_datas, test_labels = data_process.get_batch_of_data(
            image_lists, -1, FLAGS.images_dir, "test", resized_width, resized_height)

    test_loss1, test_acc1 = model.evaluate(test_datas,test_labels)

    print("Test accuracy:{0:.4f}, test loss:{1:.4f}".format(test_acc1, test_loss1))

    model.load_weights(os.path.join(FLAGS.path, "save_model", "CNN.h5df"))
    test_loss2, test_acc2 = model.evaluate(test_datas, test_labels)

    time_elapsed = time.time() - since

    print("Test accuracy with best validation =  {}".format(test_acc2 * 100))
    print("Final test accuracy =  {}".format(test_acc1 * 100))
    print("Total Model Runtime: {}min, {:0.2f}sec".format(int(time_elapsed // 60), time_elapsed % 60))

    with open(os.path.join(FLAGS.path, "results/results/results.txt"), "w") as f:
        f.write("Test accuracy with best validation:" + str(test_acc2) + "\n")
        f.write("Final test accuracy: " + str(test_acc1) + "\n")
        f.write("Total Model Runtime: " + str(int(time_elapsed // 60)) + "min," + str(time_elapsed % 60) + "sec")

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
        help="How many images to train on at a time."
    )
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
