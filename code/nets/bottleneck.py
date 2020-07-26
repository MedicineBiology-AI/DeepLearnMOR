# Modified from
# DOI: http://dx.doi.org/10.17632/rscbjbr9sj.2#file-b267c884-6b3b-43ff-a992-57960f740d0f
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
import h5py
import os

from tensorflow.python.platform import gfile
from nets import utils

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1

def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category, model_name):
  return utils.get_image_path(image_lists, label_name, index, bottleneck_dir, category) +"_"+ model_name+".h5"

# Store bottleneck values
def store_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,model_name,
                      jpeg_data_tensor, decoded_image_tensor,
                      resized_input_tensor, bottleneck_tensor,model=None):
  num_bottlenecks = 0

  if not os.path.exists(bottleneck_dir):
      os.makedirs(bottleneck_dir)

  for label_name, label_lists in image_lists.items():
    for category in ["training", "testing", "validation"]:
      category_list = label_lists[category]
      for index in range(len(category_list)):
        get_bottleneck(
          sess, image_lists, label_name, index, image_dir, category,
          bottleneck_dir, model_name, jpeg_data_tensor, decoded_image_tensor,
          resized_input_tensor, bottleneck_tensor,model=model)

        num_bottlenecks += 1
        if num_bottlenecks % 100 == 0:
          tf.logging.info("{} bottleneck files created.".format(num_bottlenecks))

# Calculate bottleneck values for image
def get_bottleneck(sess, image_lists, label_name, index, image_dir,
                   category, bottleneck_dir, model_name, jpeg_data_tensor,
                   decoded_image_tensor, resized_input_tensor,
                   bottleneck_tensor,model=None):
    label_lists = image_lists[label_name]
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                                bottleneck_dir, category,model_name)
    if not os.path.exists(bottleneck_path):
        create_bottleneck(
            sess, bottleneck_path, image_lists, label_name, index,
            image_dir, category, jpeg_data_tensor,
            decoded_image_tensor, resized_input_tensor,
            bottleneck_tensor,model=model)

    with h5py.File(bottleneck_path, "r") as bottleneck_file:
        bottleneck_string = np.array(bottleneck_file['bottleneck'])

    # if loaded or created bottleneck is invalid, recreate
    try:
        bottleneck_values = bottleneck_string
    except ValueError:
        tf.logging.warning("Error reading bottleneck, recreating bottleneck")
        create_bottleneck(
            sess, bottleneck_path, image_lists, label_name, index,
            image_dir, category, jpeg_data_tensor,
            decoded_image_tensor, resized_input_tensor,
            bottleneck_tensor,model=model)

        with h5py.File(bottleneck_path, "r") as bottleneck_file:
            bottleneck_string = np.array(bottleneck_file['bottleneck'])

        bottleneck_values = bottleneck_string
    return bottleneck_values

# Creates a bottleneck file
def create_bottleneck(sess,bottleneck_path, image_lists, label_name, index,
                      image_dir, category, jpeg_data_tensor,
                      decoded_image_tensor, resized_input_tensor,
                      bottleneck_tensor,model=None):
    tf.logging.info("Creating Bottleneck at {}".format(bottleneck_path))
    image_path = utils.get_image_path(image_lists, label_name, index,
                                      image_dir, category)
    if not gfile.Exists(image_path):
        tf.logging.fatal("File does not exist {}".format(image_path))
    image_data = gfile.FastGFile(image_path, "rb").read()
    try:
        if(model==None):
            bottleneck_values = run_bottleneck(
                sess, image_data, jpeg_data_tensor, decoded_image_tensor,
                resized_input_tensor, bottleneck_tensor)
        else:
            bottleneck_values = run_densenet_bottleneck(
                sess, image_data, jpeg_data_tensor, decoded_image_tensor,
                bottleneck_tensor, model)
    except Exception as e:
        raise RuntimeError("Error bottlenecking {}\n{}".format(image_path, str(e)))

    bottleneck_directory = "/".join(bottleneck_path.split("\\")[:-1])

    if not os.path.exists(bottleneck_directory):
        os.makedirs(bottleneck_directory)

    with h5py.File(bottleneck_path, "w") as bottleneck_file:
        bottleneck_file['bottleneck'] = bottleneck_values

# Run pre-train model on an image to generate bottleneck values
def run_bottleneck(sess, image_data, image_data_tensor,
                            decoded_image_tensor, resized_input_tensor,
                            bottleneck_tensor):
  # Preprocess input images
  resized_input_values = sess.run(decoded_image_tensor,
                                  {image_data_tensor: image_data})
  # Run resize datas through the network.
  bottleneck_values = sess.run(bottleneck_tensor,
                               {resized_input_tensor: resized_input_values})

  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values

# Run densenet on an image to generate bottleneck values
def run_densenet_bottleneck(sess, image_data, image_data_tensor,
                            decoded_image_tensor,
                            bottleneck_tensor, model):
  # Preprocess input images
  resized_input_values = sess.run(decoded_image_tensor,
                                    {image_data_tensor: image_data})
  # Run resize datas through the network.
  outputs = model.predict(resized_input_values)

  bottleneck_values = sess.run(bottleneck_tensor, {bottleneck_tensor: outputs})
  bottleneck_values = np.squeeze(bottleneck_values)

  return bottleneck_values

# Returns a batch of the bottlenecks from storage
def get_batch_of_bottlenecks(sess, image_lists, batch_size, category,
                                  bottleneck_dir, image_dir, model_name, jpeg_data_tensor,
                                  decoded_image_tensor, resized_input_tensor,
                                  bottleneck_tensor):
  class_count = len(image_lists.keys())
  bottlenecks = []
  labels = []
  if batch_size >= 0:
    # Fetch a random sample of bottlenecks.
    # Used for training set.
    for i in range(batch_size):
      label_index = random.randrange(class_count)
      label_name = list(image_lists.keys())[label_index]
      image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
      bottleneck = get_bottleneck(
        sess, image_lists, label_name, image_index, image_dir, category,
        bottleneck_dir, model_name, jpeg_data_tensor, decoded_image_tensor,
        resized_input_tensor, bottleneck_tensor)
      ground_truth = np.zeros(class_count, dtype=np.float32)
      ground_truth[label_index] = 1.0
      bottlenecks.append(bottleneck)
      labels.append(ground_truth)
  else:
    # Fetch all bottlenecks
    # Used for validation set and test set.
    for label_index, label_name in enumerate(image_lists.keys()):
      for image_index, image_name in enumerate(
          image_lists[label_name][category]):
        bottleneck = get_bottleneck(
          sess, image_lists, label_name, image_index, image_dir, category,
          bottleneck_dir, model_name, jpeg_data_tensor, decoded_image_tensor,
          resized_input_tensor, bottleneck_tensor)
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        labels.append(ground_truth)
  return bottlenecks, labels