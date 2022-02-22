import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import cv2

AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 32
CLASS_NAME = ["NRDR", "RDR"]

train_images_path = "/Users/Pankai/Downloads/DLlab/IDRID_dataset/images/train"
train_labels_path = "/Users/Pankai/Downloads/DLlab/IDRID_dataset/labels/train.csv"
test_images_path = "/Users/Pankai/Downloads/DLlab/IDRID_dataset/images/test"
test_labels_path = "/Users/Pankai/Downloads/DLlab/IDRID_dataset/labels/test.csv"

def preprocess_image_peter(path):
    """
    Take original image and crop it as good as possible from left & right.
    Afterwards resize it to square (3400x3400 -> often results had about that size)
    Args:
        image (np.array): input non-square image with many black pixels at sides
    Returns:
        np.array : square image (3400x3400) with less black pixels at sides
    """
    image = Image.open(path.numpy())
    image = tf.constant(tf.keras.preprocessing.image.img_to_array(image)).numpy()
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

    return cv2.resize(image, (3400, 3400))

def prepare_dataset(image_path, label_path):
    image_root = pathlib.Path(image_path)
    all_image_paths = list(image_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]
    images_ds = tf.data.Dataset.from_tensor_slices(all_image_paths) \
        .map(lambda x: tf.py_function(preprocess_image_peter, [x], Tout=tf.float32), num_parallel_calls=AUTO)
    # in graph mode,can't use path.numpy() use tf.py_function to define function outside of graph,
    # Tout=tf.float32 indicates the output of map_function

    df = pd.read_csv(label_path)
    train_labels = np.array(df["Retinopathy grade"])
    train_labels[train_labels < 2] = 0
    train_labels[train_labels > 1] = 1
    labels_ds = tf.data.Dataset.from_tensor_slices(train_labels)

    ds = tf.data.Dataset.zip((images_ds, labels_ds))
    return ds


# Resampling use tf.data.Dataset.random(...), don't need shuffle function any more,because of the repeat()
# dataset is infinite long,in model.fit should difine steps_per_epochs
train_ds = prepare_dataset(train_images_path, train_labels_path)
zeros_ds = train_ds.filter(lambda image, label: label == 0).repeat()
ones_ds = train_ds.filter(lambda features, label: label == 1).repeat()

# this balanced_train_ds will be put into model, the augment function can be mapped here
balanced_train_ds = tf.data.experimental.sample_from_datasets(
    [zeros_ds, ones_ds], [0.5, 0.5]).batch(BATCH_SIZE).prefetch(AUTO)

test_ds = prepare_dataset(test_images_path, test_labels_path).batch(BATCH_SIZE).prefetch(AUTO)

"""for features, labels in balanced_train_ds.take(10):
    print(labels.numpy())
"""



def show_image(ds):
    image, label = next(iter(ds))
    plt.figure(figsize=(20, 20))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image[n])
        plt.title(CLASS_NAME[label[n]])
        plt.axis("off")
    plt.show()


show_image(balanced_train_ds)
show_image(test_ds)


