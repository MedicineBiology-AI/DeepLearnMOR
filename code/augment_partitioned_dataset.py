#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2020/06/20 21:16
@Author : Jiying Li
@File : augment_partitioned_dataset.py
@Software: PyCharm
"""

import argparse
import cv2
import time
import os

from os.path import join


def main(_):
    partitioned_dataset_path = args.partitioned_dataset_dir_path

    # step 1
    # Create the dir structure for augmentedDataset:
    # AugmentedDataset --| train --| | chlo_normal | chlo_enlarged | ... |
    #                  --| val   --| | chlo_normal | chlo_enlarged | ... |
    #                  --| test  --| | chlo_normal | chlo_enlarged | ... |
    create_augmented_dataset_dirs(partitioned_dataset_path)

    # step 2. Iterate all images in PartitionedDataset, split into 4 and save in the corresponding dirs inside
    # "AugmentedDataset" directory.
    # By design, "PartitionedDataset" and "AugmentedDataset" directories are on the same level under the root dir.
    split_by4(partitioned_dataset_path)

    # step 3. Iterate all images in "AugmentedDataset/train" directory. Augment by 8 time via rotation and flipping.
    trainSetPath = partitioned_dataset_path.replace("PartitionedDataset", r"AugmentedDataset\train")
    augment_by8(trainSetPath)


def create_augmented_dataset_dirs(partitioned_dataset_path):
    """ Create the directory structure for augmented dataset
    Args:
      partitioned_dataset_path: absolute path of the partitioned dataset.
    Return:
      none
    Example: create_augmented_dataset_dirs(r"c:/DeepOrganelle/PartitionedDataset")
    """
    augmented_dataset_path = partitioned_dataset_path.replace("PartitionedDataset", "AugmentedDataset")
    categories = ["chlo_enlarged", "chlo_normal", "mito_elongated", "mito_normal", "pex_elongated", "pex_normal"]
    partitions = ["train", "val", "test"]

    for partition in partitions:
        for category in categories:
            augmentation_category_destination_path = join(join(augmented_dataset_path, partition), category)
            if not os.path.exists(augmentation_category_destination_path):
                os.makedirs(augmentation_category_destination_path)


def split_by4(dir_path):
    """ Split an image to 4 parts for all the image in the passed-in dir. Print out the duration.
    Args:
      dir_path: absolute path of partitioned dataset.
    Return:
      none
    Example: split_by4(r"c:/DeepOrganelle/PartitionedDataset")
    """
    all_imgs = get_all_imgs_in_dir(dir_path)
    start = time.time()
    for img in all_imgs:
        quarter_splitter(img)
    end = time.time()
    print(f"split_by4 takes: {end - start} seconds.")


def augment_by8(dir_path):
    """ augment an image to eight images for all the images in the passed-in dir.
    Args:
      dir_path: absolute path of train set in the augmented dataset.
    Return:
      none
    Example: augment_by8(r"c:/DeepOrganelle/AugmentedDataset/train")
    """
    all_imgs = get_all_imgs_in_dir(dir_path)
    start = time.time()
    for img in all_imgs:
        augment_img_by8(img)
    end = time.time()
    print(f"augment_by8 takes: {end - start} seconds.")


def get_all_imgs_in_dir(dir_path):
    """ Get all the jpg images path in a directory
    Args:
      dir_path: absolute path of a directory.
    Return:
      none
    Example: get_all_imgs_in_dir(r"c:/DeepOrganelle/PartitionedDataset/train")
    """
    all_imgs = [os.path.join(dir_path, filename) for dir_path, sub_dir_names, filenames in os.walk(dir_path)
                for filename in filenames if os.path.splitext(filename)[1] == '.jpg']
    return all_imgs


def quarter_splitter(img_path):
    """ Split an image to 4 parts and save all four in the corresponding augmented dataset dirs.
    Args:
      img_path: absolute path of an image in partitioned dataset.
    Return:
      none
    Example: quarter_splitter(r"c:/DeepOrganelle/PartitionedDataset/train/chlo_enlarged/123.jpg")
    """
    img = cv2.imread(img_path)
    img_height = img.shape[0]
    img_width = img.shape[1]
    quarter_height_interval = int(img_height / 2)
    quarter_width_interval = int(img_width / 2)

    # Use quarter_id, following as appendix to the quarter image
    # -----------
    # | 11 | 21 |
    # -----------
    # | 12 | 22 |
    # -----------

    for quarter_height_start in range(0, img_height, quarter_height_interval):
        for quarter_width_start in range(0, img_width, quarter_width_interval):
            quarter_height_end = quarter_height_start + quarter_height_interval
            quarter_width_end = quarter_width_start + quarter_width_interval
            quarter = img[quarter_height_start: quarter_height_end, quarter_width_start: quarter_width_end, :]

            quarter_height_id = int(quarter_height_end / quarter_height_interval)
            quarter_width_id = int(quarter_width_end / quarter_width_interval)

            quarter_id = f"{quarter_height_id}{quarter_width_id}"
            quarter_path = __get_quarter_image_path_from_partitioned_image_path(img_path, quarter_id)

            cv2.imwrite(quarter_path, quarter)


def augment_img_by8(quarter_image_path):
    """ augment an image to eight images via rotation and flip
    Args:
      quarter_image_path: absolute path of a split image.
    Return:
      none
    Example: augment_img_by8(r"c:/DeepOrganelle/AugmentedDataset/train/chlo_enlarged/123_quarterId=11.jpg")
    """
    # read base img, and do the rotation
    img1 = cv2.imread(quarter_image_path)
    img2 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
    img3 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
    img4 = cv2.rotate(img3, cv2.ROTATE_90_CLOCKWISE)

    # With img1-4, do the vertical flip as following:
    # 1->5
    # 2->6
    # 3->7
    # 4->8
    img5 = cv2.flip(img1, 0)
    img6 = cv2.flip(img2, 0)
    img7 = cv2.flip(img3, 0)
    img8 = cv2.flip(img4, 0)

    # Assign augment_id to augmented images.
    # Use the input img as the base image and assign augment_id as 1.
    # images with augment_id 2, 3, 4 are generated by 3 consecutive 90 degree clockwise rotation applied on base image.
    # images with augment_id 5, 6, 7, 8 are generated by applying vertical flip to images with augment_id 1, 2, 3, 4.
    # -----------------
    # | 1 | 2 | 3 | 4 |
    # -----------------
    #   ||
    #   ||  vertical flip
    #   \/
    # -----------------
    # | 5 | 6 | 7 | 8 |
    # -----------------

    # now all the images are ready, need to give it a name.
    # for img1 only rename it.
    # for img2-8 just create and save it.

    img1_path = __append_augment_id_to_image_name(quarter_image_path, 1)
    os.rename(quarter_image_path, img1_path)

    img_list = [img2, img3, img4, img5, img6, img7, img8]
    for augment_id in range(2, 9):
        img_index = augment_id - 2
        save_path = __append_augment_id_to_image_name(quarter_image_path, augment_id)
        cv2.imwrite(save_path, img_list[img_index])


def __get_quarter_image_path_from_partitioned_image_path(partitioned_image_path, quarter_id):
    """ split an image to 4 parts and save all four
    Args:
      partitioned_image_path: path of a partitioned image
      quarter_id: quarter id
    Return:
      none
    Example: __get_quarter_image_path_from_partitioned_image_path(r"c:/DeepOrganelle/PartitionedDataset/train/chlo_enlarged/123.jpg", 11)
    """
    augmented_img_path_prefix = partitioned_image_path.replace("PartitionedDataset", "AugmentedDataset")
    append_index = augmented_img_path_prefix.rfind(".")
    return augmented_img_path_prefix[0:append_index] + "_quarterId=" + quarter_id + augmented_img_path_prefix[append_index:]


def __append_augment_id_to_image_name(path, augment_id):
    """ append augmentId after the image name
    Args:
      path: path of an image
      augment_id: augment id
    Return:
      none
    Example: __append_augment_id_to_image_name(r"c:/DeepOrganelle/AugmentedDataset/train/chlo_enlarged/123.jpg", 1)
    """
    append_index = path.rfind(".")
    return path[0:append_index] + "_augmentId=" + str(augment_id) + path[append_index:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image processing unit.",
        epilog=r"Example: python augment_partitioned_dataset.py -p "
               r"C:\Users\laser\Desktop\PartitionedDataset"
    )
    parser.add_argument(
        "-p",
        "--partitioned-dataset-dir-path",
        type=str,
        help=r"Your partitioned dataset dir path. eg: C:\Users\laser\Desktop\PartitionedDataset")
    args = parser.parse_args()
    main(args)
