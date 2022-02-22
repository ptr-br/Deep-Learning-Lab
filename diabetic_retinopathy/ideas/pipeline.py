import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 32
CLASS_NAME = ["NRDR", "RDR"]

train_images_path = "/Users/Pankai/Downloads/DLlab/IDRID_dataset/images/train"
train_labels_path = "/Users/Pankai/Downloads/DLlab/IDRID_dataset/labels/train.csv"
test_images_path = "/Users/Pankai/Downloads/DLlab/IDRID_dataset/images/test"
test_labels_path = "/Users/Pankai/Downloads/DLlab/IDRID_dataset/labels/test.csv"

rng = tf.random.Generator.from_seed(123)


def random_apply(image, func, p=0.5):
    if tf.random.uniform([]) < p:
        image = func(image)
    else:
        image = image
    return image


def color_series(image):
    image = tf.image.stateless_random_brightness(image, 0.2, seed=rng.make_seeds(2)[0])
    image = tf.image.stateless_random_contrast(image, 0.1, 0.3, seed=rng.make_seeds(2)[0])
    image = tf.clip_by_value(image, 0, 1)
    return image


def resize_crop(image):
    img_height, img_width = tf.shape(image)[0], tf.shape(image)[1]
    image = tf.image.stateless_random_flip_left_right(image, seed=rng.make_seeds(2)[0])
    image = tf.image.stateless_random_crop(image, (200, 200, 3), seed=rng.make_seeds(2)[0])
    image = tf.image.resize(image, [img_height, img_width])
    return image


def preprocess_image(path):
    image = Image.open(path.numpy())
    image = tf.constant(tf.keras.preprocessing.image.img_to_array(image))
    image = tf.image.pad_to_bounding_box(image, 890, 340, 4628, 4628)
    image = tf.image.central_crop(image, 0.75)
    image = tf.image.resize(image, [256, 256])
    image /= 255.0

    image = random_apply(image, color_series)
    image = random_apply(image, resize_crop)

    return image


def prepare_dataset(image_path, label_path):
    image_root = pathlib.Path(image_path)
    all_image_paths = list(image_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]
    images_ds = tf.data.Dataset.from_tensor_slices(all_image_paths) \
        .map(lambda x: tf.py_function(preprocess_image, [x], Tout=tf.float32), num_parallel_calls=AUTO)
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


