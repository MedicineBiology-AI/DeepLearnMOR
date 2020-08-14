from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from nets import utils

from nets.mobilenet import mobilenet_v2
from nets.vgg import vgg
from nets.resnet import resnet_v2
from nets.inception import inception_v3


# Return dict of model info
def get_model_config():
    input_width = 299
    input_height = 299
    input_depth = 3
    input_mean = 128.0
    input_std = 128.0

    return {
        "input_width": input_width,
        "input_height": input_height,
        "input_depth": input_depth,
        "input_mean": input_mean,
        "input_std": input_std,
    }


# Returns tensors to feed jpeg data into
def decode_jpeg(input_width, input_height, input_depth, input_mean, input_std):
    jpeg_data = tf.placeholder(tf.string, name="DecodeJPEGInput")
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                             resize_shape_as_int)
    offset_image = tf.subtract(resized_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    return jpeg_data, mul_image


# Return graph of network and bottleneck tensor
# Load Inception v3 model
def load_inception_v3(model_dir):
    inception_url = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"
    bottleneck_tensor_name = "pool_3/_reshape:0"
    resized_input_tensor_name = "Mul:0"
    model_file_name = "classify_image_graph_def.pb"

    utils.download_inception_weights(inception_url, model_dir)

    with tf.Graph().as_default() as graph:
        model_path = os.path.join(model_dir, model_file_name)
        with gfile.FastGFile(model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
                graph_def,
                name="",
                return_elements=[
                    bottleneck_tensor_name,
                    resized_input_tensor_name,
                ]))

    bottleneck_tensor_size = 2048

    return graph, bottleneck_tensor, resized_input_tensor, bottleneck_tensor_size


# Load VGG 16 model
def load_vgg_16(model_dir, sess):
    model_file_name = "vgg_16_2016_08_28/vgg_16.ckpt"
    model_path = os.path.join(model_dir, model_file_name)

    resized_input_tensor = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    with tf.contrib.slim.arg_scope(vgg.vgg_arg_scope()):
        bottleneck_tensor, _ = vgg.vgg_16(
            resized_input_tensor, num_classes=None, global_pool=True)

    variable_restore_op = tf.contrib.slim.assign_from_checkpoint_fn(
        model_path,
        tf.contrib.slim.get_trainable_variables(),
        ignore_missing_vars=True)
    variable_restore_op(sess)

    bottleneck_tensor = tf.squeeze(bottleneck_tensor, axis=[1, 2])
    bottleneck_tensor_size = 4096

    return bottleneck_tensor, resized_input_tensor, bottleneck_tensor_size


# Load ResNet v2 50 model
def load_resnet_v2_50(model_dir, sess):
    model_file_name = "resnet_v2_50_2017_04_14/resnet_v2_50.ckpt"
    model_path = os.path.join(model_dir, model_file_name)

    resized_input_tensor = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope()):
        bottleneck_tensor, _ = resnet_v2.resnet_v2_50(
            resized_input_tensor, num_classes=None, global_pool=True)

    variable_restore_op = tf.contrib.slim.assign_from_checkpoint_fn(
        model_path,
        tf.contrib.slim.get_trainable_variables(),
        ignore_missing_vars=True)
    variable_restore_op(sess)

    bottleneck_tensor = tf.squeeze(bottleneck_tensor, axis=[1, 2])
    bottleneck_tensor_size = 2048

    return bottleneck_tensor, resized_input_tensor, bottleneck_tensor_size


# Load ResNet v2 101 model
def load_resnet_v2_101(model_dir, sess):
    model_file_name = "resnet_v2_101_2017_04_14/resnet_v2_101.ckpt"
    model_path = os.path.join(model_dir, model_file_name)

    resized_input_tensor = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope()):
        bottleneck_tensor, _ = resnet_v2.resnet_v2_101(
            resized_input_tensor, num_classes=None, global_pool=True)

    variable_restore_op = tf.contrib.slim.assign_from_checkpoint_fn(
        model_path,
        tf.contrib.slim.get_trainable_variables(),
        ignore_missing_vars=True)
    variable_restore_op(sess)

    bottleneck_tensor = tf.squeeze(bottleneck_tensor, axis=[1, 2])
    bottleneck_tensor_size = 2048

    return bottleneck_tensor, resized_input_tensor, bottleneck_tensor_size


# Load ResNet v2 152 model
def load_resnet_v2_152(model_dir, sess):
    model_file_name = "resnet_v2_152_2017_04_14/resnet_v2_152.ckpt"
    model_path = os.path.join(model_dir, model_file_name)

    resized_input_tensor = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope()):
        bottleneck_tensor, _ = resnet_v2.resnet_v2_152(
            resized_input_tensor, num_classes=None, global_pool=True)

    variable_restore_op = tf.contrib.slim.assign_from_checkpoint_fn(
        model_path,
        tf.contrib.slim.get_trainable_variables(),
        ignore_missing_vars=True)
    variable_restore_op(sess)

    bottleneck_tensor = tf.squeeze(bottleneck_tensor, axis=[1, 2])
    bottleneck_tensor_size = 2048

    return bottleneck_tensor, resized_input_tensor, bottleneck_tensor_size


# Load MobileNet v2 model
def load_mobilenet_v2(model_dir, sess):
    model_file_name = "mobilenet_v2_1.4_224/mobilenet_v2_1.4_224.ckpt"
    model_path = os.path.join(model_dir, model_file_name)

    resized_input_tensor = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
        bottleneck_tensor, _ = mobilenet_v2.mobilenet(
            resized_input_tensor, num_classes=None, depth_multiplier=1.4)

    variable_restore_op = tf.contrib.slim.assign_from_checkpoint_fn(
        model_path,
        tf.contrib.slim.get_trainable_variables(),
        ignore_missing_vars=True)
    variable_restore_op(sess)

    # bottleneck_tensor = tf.squeeze(bottleneck_tensor, axis=[1, 2])
    bottleneck_tensor_size = 1792

    return bottleneck_tensor, resized_input_tensor, bottleneck_tensor_size


# Load DenseNet 121 model
def load_densenet_121(model_dir):
    model_file_name = "densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5"
    model_path = os.path.join(model_dir, model_file_name)

    with tf.name_scope("DenseNet"):
        model = tf.keras.applications.densenet.DenseNet121(include_top=False, weights=None, pooling='avg')
        model.load_weights(model_path)

        bottleneck_tensor_size = 1024
        bottleneck_tensor = tf.placeholder(tf.float32, [None, bottleneck_tensor_size])
        resized_input_tensor = None

    return model, bottleneck_tensor, resized_input_tensor, bottleneck_tensor_size


# Load DenseNet 169 model
def load_densenet_169(model_dir):
    model_file_name = "densenet/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5"
    model_path = os.path.join(model_dir, model_file_name)

    with tf.name_scope("DenseNet"):
        model = tf.keras.applications.densenet.DenseNet169(include_top=False, weights=None, pooling='avg')
        model.load_weights(model_path)

        bottleneck_tensor_size = 1664
        bottleneck_tensor = tf.placeholder(tf.float32, [None, bottleneck_tensor_size])
        resized_input_tensor = None

    return model, bottleneck_tensor, resized_input_tensor, bottleneck_tensor_size


# Load DenseNet 201 model
def load_densenet_201(model_dir):
    model_file_name = "densenet/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5"
    model_path = os.path.join(model_dir, model_file_name)

    with tf.name_scope("DenseNet"):
        model = tf.keras.applications.densenet.DenseNet201(include_top=False, weights=None, pooling='avg')
        model.load_weights(model_path)

        bottleneck_tensor_size = 1920
        bottleneck_tensor = tf.placeholder(tf.float32, [None, bottleneck_tensor_size])
        resized_input_tensor = None

    return model, bottleneck_tensor, resized_input_tensor, bottleneck_tensor_size
