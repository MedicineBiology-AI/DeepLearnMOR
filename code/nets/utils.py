# Modified from
# DOI: http://dx.doi.org/10.17632/rscbjbr9sj.2#file-b267c884-6b3b-43ff-a992-57960f740d0f
import tensorflow as tf
import tarfile
import sys
import os

from glob import glob
from six.moves import urllib

# Return a sorted list of image files at image_dir
def get_image_files(image_dir):
  fs = glob("{}/*.jpg".format(image_dir))
  fs = [os.path.basename(filename) for filename in fs]
  return sorted(fs)

# Download inception model if not already at 'inception_url'
def download_inception_weights(inception_url, dest_dir):
  if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

  filename = inception_url.split("/")[-1]
  filepath = os.path.join(dest_dir, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write("\r>> Downloading {} {:0.1f}".format(
        filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(inception_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    tf.logging.info("Successfully downloaded", filename, statinfo.st_size,"bytes.")
    tarfile.open(filepath, "r:gz").extractall(dest_dir)

def create_image_lists(image_dir):
  result = {}

  training_images = []
  testing_images = []
  validation_images = []

  for category in ["train", "test", "val"]:
    category_path = os.path.join(image_dir, category)
    try:
      class_labels = next(os.walk(category_path))[1]
      class_labels.sort()
    except StopIteration:
      sys.exit("ERROR: Missing either train/test/val folders in image_dir")
    for cls in class_labels:
      class_path = os.path.join(category_path, cls)
      if category == "train":
        training_images.append(get_image_files(class_path))
      if category == "test":
        testing_images.append(get_image_files(class_path))
      if category == "val":
        validation_images.append(get_image_files(class_path))

  for cls in class_labels:
    result[cls] = {
      "training": training_images[class_labels.index(cls)],
      "testing": testing_images[class_labels.index(cls)],
      "validation": validation_images[class_labels.index(cls)],
    }
  return result

# Return a path to an image with the given label at the given index
def get_image_path(image_lists, label_name, index, image_dir, category):
  if label_name not in image_lists:
    tf.logging.fatal("Label does not exist %s.", label_name)
  label_lists = image_lists[label_name]
  if category not in label_lists:
    tf.logging.fatal("Category does not exist %s.", category)
  category_list = label_lists[category]
  if not category_list:
    tf.logging.fatal("Label %s has no images in the category %s.", label_name, category)
  mod_index = index % len(category_list)
  base_name = category_list[mod_index]

  if("train" in category):
    full_path = os.path.join(image_dir, "train", label_name, base_name)
  elif("test" in category):
    full_path = os.path.join(image_dir, "test", label_name, base_name)
  elif("val" in category):
    full_path = os.path.join(image_dir, "val", label_name, base_name)

  return full_path