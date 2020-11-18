import numpy as np
import tensorflow as tf
import os
import sys
import h5py
import cv2
import random
from glob import glob


def get_image_files(image_dir):
    fs = glob("{}/*.jpeg".format(image_dir))
    fs = [os.path.basename(filename) for filename in fs]
    return sorted(fs)


def create_image_lists(image_dir):
    result = {}

    training_images = []
    testing_images = []
    validation_images = []

    for category in ["train", "test", "val"]:
        category_path = os.path.join(image_dir, category)
        try:
            bins = next(os.walk(category_path))[1]
            bins.sort()
        except StopIteration:
            sys.exit("ERROR: Missing either train/test/val folders in image_dir")
        for diagnosis in bins:
            bin_path = os.path.join(category_path, diagnosis)
            if category == "train":
                training_images.append(get_image_files(bin_path))
            if category == "test":
                testing_images.append(get_image_files(bin_path))
            if category == "val":
                validation_images.append(get_image_files(bin_path))

    for diagnosis in bins:
        result[diagnosis] = {
            "train": training_images[bins.index(diagnosis)],
            "test": testing_images[bins.index(diagnosis)],
            "val": validation_images[bins.index(diagnosis)],
        }
    return result


def image_normalization(image_data, resized_width, resized_height):
    resized_image = cv2.resize(image_data, (resized_width, resized_height))
    image_stand = resized_image / 255.0
    return image_stand


def get_batch_of_data(image_lists, batch_size, image_dir, category, resized_width, resized_height):
    class_count = len(image_lists.keys())
    image_datas = []
    labels = []
    if (batch_size > 0):
        for i in range(batch_size):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            images = image_lists[label_name][category]
            image_index = random.randrange(len(images))
            image_name = images[image_index]
            image_path = os.path.join(image_dir, category, label_name, image_name)
            image_data = cv2.imread(image_path)
            image_datas.append(image_normalization(image_data, resized_width, resized_height))
            ground_truth = np.repeat(-1.0, 6)
            ground_truth[label_index] = 1.0
            labels.append(ground_truth)

    else:
        for label_index, label_name in enumerate(image_lists.keys()):
            for image_index, image_name in enumerate(image_lists[label_name][category]):
                image_path = os.path.join(image_dir, category, label_name, image_name)
                image_data = cv2.imread(image_path)
                image_datas.append(image_normalization(
                    image_data, resized_width, resized_height))
                ground_truth = np.repeat(-1.0, 6)
                ground_truth[label_index] = 1.0
                labels.append(ground_truth)

    image_datas = np.array(image_datas)
    image_datas = image_datas.reshape(image_datas.shape[0], -1)
    labels = np.array(labels)

    return image_datas, labels


if __name__ == "__main__":
    images_dir = "../../DeepOrganelleDataset"
    image_lists = create_image_lists(images_dir)
    train_datas, train_labels = get_batch_of_data(
        image_lists, 128, images_dir, "train", 128, 128)
    print(train_datas.shape)
    print(train_labels.shape)
    print(train_labels)
