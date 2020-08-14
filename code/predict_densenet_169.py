from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import h5py
import argparse
import sys
import os

from nets import utils
from nets import pretrain_model
from nets import train
from nets import bottleneck

'''
os.environ["CUDA_VISIBLE_DEVICES"]="0"
'''
tf.reset_default_graph()

FLAGS = None

# Read in labels
def load_labels(filename):
    return [line.rstrip() for line in tf.gfile.GFile(filename)]

# Load saved graph
def load_graph(filename):
    with tf.gfile.FastGFile(filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")

def predict(images_dir, model_dir, bottlenecks_dir, input_layer_name, output_layer_name,model_name):
    model_config = pretrain_model.get_model_config()

    image_lists = utils.create_image_lists(images_dir)

    with tf.Session() as sess:
        jpeg_data, decoded_image = pretrain_model.decode_jpeg(
            model_config["input_width"], model_config["input_height"],
            model_config["input_depth"], model_config["input_mean"],
            model_config["input_std"])

        densenet_model, bottlenecks, resized_image, bottlenecks_size = pretrain_model.load_densenet_169(
            model_dir)

        load_graph(FLAGS.graph)

        test_bottlenecks, test_labels = (
            bottleneck.get_batch_of_bottlenecks(
                sess, image_lists, -1, "testing",
                bottlenecks_dir, images_dir, model_name, jpeg_data,
                decoded_image, resized_image, bottlenecks))

        softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
        bottlenecks_input = sess.graph.get_tensor_by_name(input_layer_name)

        prediction = sess.run(softmax_tensor, feed_dict={bottlenecks_input: test_bottlenecks})
        # print(sess.run(tf.shape(prediction)))

        y_pred = tf.placeholder(tf.float32, [None, len(image_lists.keys())])
        y_truth = tf.placeholder(tf.float32, [None, len(image_lists.keys())])

        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_truth, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_truth, logits=y_pred))

        accuracy, loss = sess.run([acc, cross_entropy], feed_dict={y_pred: prediction, y_truth: test_labels})

        print("test accuracy: {0:.7f}, test loss: {1:.7f}".format(accuracy, loss))

        pre_data_dir = os.path.join("../predictData", model_name)

        if not os.path.exists(pre_data_dir):
            os.makedirs(pre_data_dir)

        with h5py.File(os.path.join(pre_data_dir, "prediction_and_labels.h5"), "w") as f:
            f["prediction"] = prediction
            f["truth"] = test_labels

    return 0

def main(argv):
    labels = load_labels(FLAGS.labels)

    predict(FLAGS.images_dir, FLAGS.model_dir, FLAGS.bottlenecks_dir,
            FLAGS.input_layer_name, FLAGS.output_layer_name,FLAGS.model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_dir",
        type=str,
        default="../DeepOrganelleAugmentedDataset",
        help="Images folder directory."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="densenet169",
        help="The name of the model."
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="../densenet169/retrained_graph.pb",
        help="Absolute path to graph file (.pb)")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="../pretrained",
        help="Path to pretrained weights"
    )
    parser.add_argument(
        "--bottlenecks_dir",
        type=str,
        default="../densenet169/bottlenecks",
        help="Path to store bottleneck layer values."
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="../densenet169/output_labels.txt",
        help="Absolute path to labels file (.txt)")
    parser.add_argument(
        "--output_layer_name",
        type=str,
        default="final_result:0",
        help="Output_layer_name")
    parser.add_argument(
        "--input_layer_name",
        type=str,
        default="input/BottleneckInput:0",
        help="Input layer name")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=sys.argv[:1] + unparsed)
