import numpy as np
import os
import pandas as pd
import tensorflow as tf
import cv2
import logging
import gin
import sys

from sklearn.model_selection import train_test_split
from utils import utils_tfrecords
from sklearn.utils import resample


RANDOM_SEED = 42


def convert_binary(df):
    """
    Turns dataframe with multiple retinopathy grades into binary formart

    Args:
        df (dataframe): Retinopathy grade of dataframe (non-binary)

    Returns:
        dataframe: binary dataframe in retinopathy grades  (0/1)
    """

    df.loc[df['Retinopathy grade'] < 2, "Retinopathy grade"] = 0
    df.loc[df['Retinopathy grade'] > 1, "Retinopathy grade"] = 1

    return df


def image_example(image, label):
    """ Create the features dictionary - Adapted from tensorflow docs"""
    string_image = cv2.imencode('.jpg', image)[1].tobytes()
    feature = {
        'label': utils_tfrecords.int64_feature(label),
        'image_raw': utils_tfrecords.bytes_feature(string_image),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def resample_df(df):
    """
    Resample imbalanced dataset to obtain a balanced one

    Args:
        df (dataframe): inbalanced dataframe

    Returns:
        dataframe: balanced dataframe
    """
    # TODO: add extension for non binary regression model

    num_samples = df['Retinopathy grade'].value_counts().max()
    df_0 = df.loc[df['Retinopathy grade'] == 0]
    df_1 = df.loc[df['Retinopathy grade'] == 1]
    df_0_resampled = resample(
        df_0, replace=True, n_samples=num_samples, random_state=RANDOM_SEED)
    df_1_resampled = resample(
        df_1, replace=True, n_samples=num_samples, random_state=RANDOM_SEED)
    df = pd.concat([df_0_resampled, df_1_resampled])

    return df


def preprocess_image(image):
    """
    Take original image and crop it as good as possible from left & right.
    Afterwards resize it to square (256 x x256 -> desired size for calssififer)

    Args:
        image (np.array): input non-square image with many black pixels at sides

    Returns:
        np.array : square image (256 x 256) with less black pixels at sides
    """
    # grayscale image for easier thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # binary threshold
    _, threshold = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    # get nonzero positions (left & right)
    pos = np.nonzero(threshold)
    right_boundary = pos[1].max()
    left_boundary = pos[1].min()
    # crop image where possible (left & right)
    image = image[:, left_boundary:right_boundary]
    # computations to obtain square image (padding at desired positions)
    upper_diff = (image.shape[1] - image.shape[0]) // 2
    lower_diff = image.shape[1] - image.shape[0] - upper_diff
    image = cv2.copyMakeBorder(image, upper_diff, lower_diff, 0, 0, cv2.BORDER_CONSTANT)

    return cv2.resize(image, (256, 256))


def record_writer(df, records_path, images_path):

    with tf.io.TFRecordWriter(records_path) as writer:
        for _, row in df.iterrows():
            image = cv2.imread(images_path + row["Image name"] + ".jpg")
            preprocessed_image = preprocess_image(image)
            label = row["Retinopathy grade"]
            tf_example = image_example(preprocessed_image, label)
            writer.write(tf_example.SerializeToString())
        writer.close()


@gin.configurable
def create_tfrecords(data_dir, records_dir):

    LABELS_PATH = data_dir + "labels/"
    IMAGES_PATH = data_dir + "images/"

    # fix seed to reproduce results each run - comment this out if unwanted
    np.random.seed(RANDOM_SEED)

    # if path already exists don't create files
    if os.path.exists(records_dir):
        return 0

    # check if paths exist -> exit if there is no data in the provided path
    if not os.path.isdir(LABELS_PATH):
        logging.error(f"Path does not exist: {LABELS_PATH}")
        logging.error("Please provide path to labels!")
        sys.exit()

    # read data from csv-file
    df_train_val = pd.read_csv(LABELS_PATH + "train.csv",
                               usecols=['Image name', 'Retinopathy grade'])
    df_test = pd.read_csv(LABELS_PATH + "test.csv",
                          usecols=['Image name', 'Retinopathy grade'])

    convert_binary(df_train_val)
    convert_binary(df_test)

    # create test/validation split
    df_train, df_val = train_test_split(df_train_val, test_size=0.2)

    df_train = resample_df(df_train)
    df_val = resample_df(df_val)

    # logging number of samplesn
    logging.info("")
    logging.info(f"training samples:    {df_train.shape[0]}")
    logging.info(f"validation samples:  {df_val.shape[0]}")
    logging.info(f"test samples:        {df_test.shape[0]}")
    logging.info("")

    # check if paths exist
    if not (os.path.isdir(IMAGES_PATH + "train/") or IMAGES_PATH + "test/"):
        logging.error(f"Path does not exist: {IMAGES_PATH}train/")
        logging.error(f"Path does not exist: {IMAGES_PATH}test/")
        logging.error("Please provide path to images!")
        sys.exit()

    # create path and write record files

    os.makedirs(records_dir)

    record_writer(df_train, records_dir +
                  "train.tfrecords", IMAGES_PATH + "train/")
    record_writer(df_val, records_dir + "validation.tfrecords",
                  IMAGES_PATH + "train/")
    record_writer(df_test, records_dir +
                  "test.tfrecords", IMAGES_PATH + "test/")

    return 1
