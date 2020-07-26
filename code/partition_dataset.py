#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 21:36:04 2018

@author: Jiying Li
"""
import argparse
import os
import shutil

from os import listdir
from os.path import isfile, join
from sklearn.utils import shuffle


def main(_):
    dataset_path = args.dataset_path
    partition_ratio = {"train": 8, "val": 1, "test": 1}

    categories = ["chlo_enlarged", "chlo_normal", "mito_elongated", "mito_normal", "pex_elongated", "pex_normal"]
    for category in categories:
        partition_one_category(dataset_path, category, partition_ratio)


def partition_one_category(dataset_path, category, partition_ratio):
    """Copy and partition the dataset of one category to three sets: train, val, test.
    Args:
      dataset_path: absolute path of the raw dataset.
      category: name of category
      partition_ratio: a dictionary containing ratio of train set, val set and test set.

    Returns:
      None

    Example:
      "c:/DeepOrganelle/Dataset/chlo_enlarged" will be copied and partitioned to three folders:
        1. "c:/DeepOrganelle/PartitionedDataset/train/chlo_enlarged"
        2. "c:/DeepOrganelle/PartitionedDataset/val/chlo_enlarged"
        3. "c:/DeepOrganelle/PartitionedDataset/test/chlo_enlarged"
    """

    partitioned_dataset_path = dataset_path.replace("Dataset", "PartitionedDataset")
    category_path = join(dataset_path, category)

    # List of all the image names in one category.
    category_images = [item for item in listdir(category_path) if isfile(join(category_path, item))]
    randomized_category_images = shuffle(category_images)

    partition_range = get_partition_range_from_ratio(partition_ratio, len(randomized_category_images))
    for partition in partition_range.keys():
        partition_category_destination_path = join(join(partitioned_dataset_path, partition), category)
        if os.path.exists(partition_category_destination_path):
            shutil.rmtree(partition_category_destination_path)
        os.makedirs(partition_category_destination_path)
        for image_index in partition_range[partition]:
            image_name = randomized_category_images[image_index]
            move_image(category_path, partition_category_destination_path, image_name)


def get_partition_range_from_ratio(partition_ratio, category_image_num):
    """construct a dictionary that maps partition name to image range based on partition ratio.
    Args:
      partition_ratio: a dictionary containing ratio of train set, val set and test set.
      category_image_num: total number of images in one category.

    Returns:
      A dictionary containing index range of images for train set, val set and test set.

    Example:
      Input:
        partition_ratio = {"train": 8, "val": 1, "test": 1}
        category_image_num = 300
      output:
        partition_range = {"train": range(0, 240), "val": range(240, 270), "test": range(270, 300)}
    """

    ratio_sum = float(sum(partition_ratio.values()))
    partition_range = {}
    start = 0
    end = 0
    for partition in partition_ratio.keys():
        current_partition_size = int((partition_ratio[partition] / ratio_sum) * category_image_num)
        end += current_partition_size
        partition_range[partition] = range(start, end)
        start += current_partition_size
    return partition_range


def move_image(src_folder_path, dst_folder_path, image_name):
    """Move an image from source folder to destination folder.
    Args:
      src_folder_path: source folder path.
      dst_folder_path: destination folder path.
      image_name: image name.

    Returns:
      None
    """

    src_image_path = join(src_folder_path, image_name)
    dst_image_path = join(dst_folder_path, image_name)
    shutil.copy(src_image_path, dst_image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Partition dataset to three sets: training, validation and test set.",
        epilog=r"Example: python partition_dataset.py -d c:\repos\DeepOrganelle\Dataset")
    parser.add_argument(
        "-d",
        "--dataset-path",
        type=str,
        help=r"path of your dataset, eg: c:\repos\DeepOrganelle\Dataset")
    args = parser.parse_args()
    main(args)
