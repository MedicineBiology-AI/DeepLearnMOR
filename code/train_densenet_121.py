# Modified from
# DOI: http://dx.doi.org/10.17632/rscbjbr9sj.2#file-b267c884-6b3b-43ff-a992-57960f740d0f
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import argparse
import os
import sys
import time

from nets import utils
from nets import pretrain_model
from nets import train
from nets import bottleneck
'''
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''
tf.reset_default_graph()

FLAGS = None

def main(_):
    since = time.time()

    tf.logging.set_verbosity(tf.logging.INFO)

    # Create directories to store TensorBoard summaries
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)

    # Set model Hyperparameter
    model_config = pretrain_model.get_model_config()

    # Read the folder structure, and create lists of all the images
    image_lists = utils.create_image_lists(FLAGS.images_dir)
    class_count = len(image_lists.keys())

    if class_count == 0:
        tf.logging.error("No valid folders of images found at " + FLAGS.images_dir)
        return -1
    if class_count == 1:
        tf.logging.error("Only one valid folder of images found at " + FLAGS.images_dir)
        return -1

    # Create output_labels.txt displaying classes being trained
    with open(FLAGS.output_labels, "w") as f:
        f.write("\n".join(image_lists.keys()) + "\n")

    with tf.Session() as sess:
        # Set up the image decoding
        jpeg_data, decoded_image = pretrain_model.decode_jpeg(
            model_config["input_width"], model_config["input_height"],
            model_config["input_depth"], model_config["input_mean"],
            model_config["input_std"])

        # Load DenseNet model
        densenet_model, bottlenecks, resized_image, bottlenecks_size = pretrain_model.load_densenet_121(
            FLAGS.model_dir)

        # store pretrained model bottlenecks
        bottleneck.store_bottlenecks(
            sess, image_lists, FLAGS.images_dir, FLAGS.bottlenecks_dir, FLAGS.model_name,
            jpeg_data, decoded_image, resized_image,
            bottlenecks, model=densenet_model)

        bottlenecks = tf.stop_gradient(bottlenecks)

        global_step = tf.Variable(tf.constant(0), trainable=False)

        # Initialized final layer
        (train_step, cross_entropy, accuracy, bottlenecks_input, labels_input,
         final_result) = train.final_layer(
            len(image_lists.keys()), FLAGS.final_name, bottlenecks,
            bottlenecks_size, FLAGS.learning_rate, global_step)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + "/train",
                                             sess.graph)

        validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + "/validation")

        # Initialize all variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # Get validation bottlenecks for evaluation
        validation_bottlenecks, validation_labels = (
            bottleneck.get_batch_of_bottlenecks(
                sess, image_lists, FLAGS.validation_batch_size, "validation",
                FLAGS.bottlenecks_dir, FLAGS.images_dir, FLAGS.model_name, jpeg_data,
                decoded_image, resized_image, bottlenecks))

        # Get test bottlenecks for evaluation
        test_bottlenecks, test_labels = (
            bottleneck.get_batch_of_bottlenecks(
                sess, image_lists, FLAGS.test_batch_size, "testing",
                FLAGS.bottlenecks_dir, FLAGS.images_dir, FLAGS.model_name, jpeg_data,
                decoded_image, resized_image, bottlenecks))

        best_acc = 0.0

        for i in range(FLAGS.iterations):
            # Get training bottlenecks
            (train_bottlenecks, train_labels) = bottleneck.get_batch_of_bottlenecks(
                sess, image_lists, FLAGS.train_batch_size, "training",
                FLAGS.bottlenecks_dir, FLAGS.images_dir, FLAGS.model_name, jpeg_data,
                decoded_image, resized_image, bottlenecks)
            # Training step
            train_summary, _ = sess.run(
                [merged, train_step],
                feed_dict={bottlenecks_input: train_bottlenecks,
                           labels_input: train_labels,
                           global_step: i})
            train_writer.add_summary(train_summary, i)

            # Show evaluation based on specified frequency
            final_step = (i + 1 == FLAGS.iterations)
            if (i % FLAGS.eval_interval) == 0 or final_step:
                # Evaluation
                train_accuracy, train_loss = sess.run(
                    [accuracy, cross_entropy],
                    feed_dict={bottlenecks_input: train_bottlenecks,
                               labels_input: train_labels})

                # Run evaluation step on validation bottlenecks
                validation_summary, validation_accuracy, validation_loss = sess.run(
                    [merged, accuracy, cross_entropy],
                    feed_dict={bottlenecks_input: validation_bottlenecks,
                               labels_input: validation_labels})
                validation_writer.add_summary(validation_summary, i)

                # Save best accuracy and store model
                if validation_accuracy > best_acc:
                    best_acc = validation_accuracy

                    # Calculate the test accuracy with best validation on test bottlenecks
                    test_accuracy1, test_loss1 = sess.run(
                        [accuracy, cross_entropy],
                        feed_dict={bottlenecks_input: test_bottlenecks,
                                   labels_input: test_labels})

                    train.save_graph_to_file(sess, FLAGS.output_graph, FLAGS.final_name)
                    train.save_checkpoint_to_file(sess, FLAGS.output_checkpoint_dir)

                tf.logging.info("Iteration {}: train loss = {}, train acc = {}, val loss = {}, val acc = {}.".format(
                    i, train_loss, train_accuracy, validation_loss, validation_accuracy))

        # Calculate the final test accuracy on test bottlenecks
        test_accuracy2, test_loss2 = sess.run(
            [accuracy, cross_entropy],
            feed_dict={bottlenecks_input: test_bottlenecks,
                       labels_input: test_labels})

        tf.logging.info("Best validation accuracy = {}".format(best_acc * 100))
        tf.logging.info("Test accuracy with best validation =  {}".format(test_accuracy1 * 100))
        tf.logging.info("Final test accuracy =  {}".format(test_accuracy2 * 100))

    time_elapsed = time.time() - since

    print("Runtime: {}min, {:0.2f}sec".format(int(time_elapsed // 60), time_elapsed % 60))

    with open(os.path.join("..", FLAGS.model_name, "results.txt"), "w") as f:
        f.write("Best validation accuracy: " + str(best_acc) + "\n")
        f.write("Test accuracy with best validation: " + str(test_accuracy1) + "\n")
        f.write("Final test accuracy =  {}".format(test_accuracy2 * 100) + "\n")
        f.write("Runtime: " + str(int(time_elapsed // 60)) + "min," + str(time_elapsed % 60) + "sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_dir",
        type=str,
        default="../DeepOrganelleAugmentedDataset",
        help="Images folder directory."
    )
    parser.add_argument(
        "--output_graph",
        type=str,
        default="../densenet121/retrained_graph.pb",
        help="Output directory to save the trained graph."
    )
    parser.add_argument(
        "--output_checkpoint_dir",
        type=str,
        default="../densenet121/checkpoint",
        help="Output directory to save the trained graph."
    )
    parser.add_argument(
        "--output_labels",
        type=str,
        default="../densenet121/output_labels.txt",
        help="Directory to save the output labels."
    )
    parser.add_argument(
        "--summaries_dir",
        type=str,
        default="../densenet121/retrain_logs",
        help="Path to save summary logs for TensorBoard."
    )
    parser.add_argument(
        "--bottlenecks_dir",
        type=str,
        default="../densenet121/bottlenecks",
        help="Path to store bottlenecks layer values."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="../pretrained",
        help="Path to pretrained weights"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="densenet121",
        help="The name of the pre-trained model"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5000,
        help="The number of iterations"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=10,
        help="How often to evaluate the training results."
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=256,
        help="How many images to train on at a time."
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=-1,
        help="Number of images from test set to test on. Value of -1 will use entire test set."
    )
    parser.add_argument(
        "--validation_batch_size",
        type=int,
        default=-1,
        help="Number of images from validation set to validate on. Value of -1 will use entire validation set."
    )
    parser.add_argument(
        "--final_name",
        type=str,
        default="final_result",
        help="The name of the output classification layer"
    )
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)