# Modified from
# DOI: http://dx.doi.org/10.17632/rscbjbr9sj.2#file-b267c884-6b3b-43ff-a992-57960f740d0f
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util


# Add summaries to a tensor for TensorBoard
def add_variable_summaries(variable):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(variable)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(variable))
        tf.summary.scalar('min', tf.reduce_min(variable))
        tf.summary.histogram('histogram', variable)


# Add a newly initialized FC layer and softmax layer for training
def final_layer(class_count, final_tensor_name, bottleneck_tensor,
                bottleneck_tensor_size, learning_rate, global_step):
    with tf.name_scope("input"):
        bottleneck_input = tf.placeholder_with_default(
            bottleneck_tensor,
            shape=[None, bottleneck_tensor_size],
            name="BottleneckInput")

        labels_input = tf.placeholder(
            tf.float32,
            [None, class_count],
            name="LabelInput")

    with tf.name_scope("final_layers"):
        with tf.name_scope("weights"):
            # Initialize random weights
            initial_value = tf.truncated_normal(
                [bottleneck_tensor_size, class_count], stddev=0.001)

            layer_weights = tf.Variable(initial_value, name="final_weights")

            add_variable_summaries(layer_weights)
        with tf.name_scope("biases"):
            layer_biases = tf.Variable(tf.zeros([class_count]), name="final_biases")
            add_variable_summaries(layer_biases)
        with tf.name_scope("FC"):
            FC_layer = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram("pre_activations", FC_layer)

    final_tensor = tf.nn.softmax(FC_layer, name=final_tensor_name)
    tf.summary.histogram("activations", final_tensor)

    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_input, logits=final_tensor)
        with tf.name_scope("total"):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar("cross_entropy", cross_entropy_mean)

    with tf.name_scope("accuracy"):
        with tf.name_scope("correct_prediction"):
            correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(labels_input, 1))
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

    with tf.name_scope("train"):
        # lr=learning_rate
        lr = tf.train.exponential_decay(learning_rate, global_step, 1600, 0.1, staircase=False)
        optimizer = tf.train.AdamOptimizer(lr)
        train_step = optimizer.minimize(cross_entropy_mean)

    return (train_step, cross_entropy_mean, accuracy, bottleneck_input, labels_input,
            final_tensor)


# Save retrained model (.pb) to' output_file'
def save_graph_to_file(sess, output_file, final_tensor_name):
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, sess.graph_def, [final_tensor_name])
    with gfile.FastGFile(output_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())


# Save retrained model (.ckpt) to' output_file'
def save_checkpoint_to_file(sess, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file_path = os.path.join(output_path, 'save_model')
    saver = tf.train.Saver()
    saver.save(sess, file_path)
