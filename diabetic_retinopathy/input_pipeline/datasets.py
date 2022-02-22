import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
from utils import utils_tfrecords
from input_pipeline.preprocessing import preprocess, augment


@gin.configurable
def load(name, data_dir, server=False):
    if name == "idrid":

        # Create the dataset object from tfrecord file(s)
        ds_train_raw = tf.data.TFRecordDataset(data_dir + "train.tfrecords")
        ds_test_raw = tf.data.TFRecordDataset(data_dir + "test.tfrecords")
        ds_val_raw = tf.data.TFRecordDataset(data_dir + "validation.tfrecords")

        decoded_ds_train = ds_train_raw.map(
            utils_tfrecords.parse_and_decode_record)
        decoded_ds_test = ds_test_raw.map(
            utils_tfrecords.parse_and_decode_record)
        decoded_ds_val = ds_val_raw.map(
            utils_tfrecords.parse_and_decode_record)

        return prepare(decoded_ds_train, decoded_ds_val, decoded_ds_test, "idrid_dataset")

    elif name == "eyepacs":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'diabetic_retinopathy_detection/btgraham-300',
            split=['train', 'validation', 'test'],
            shuffle_files=True,
            with_info=True,
            data_dir=data_dir
        )

        def _preprocess(img_label_dict):
            return img_label_dict['image'], img_label_dict['label']

        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "mnist":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train[:90%]', 'train[90%:]', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            data_dir=data_dir
        )

        return prepare(ds_train, ds_val, ds_test, ds_info)

    else:
        raise ValueError


@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching, shuffle_buffer=100):
    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.map(
        augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if isinstance(ds_info, str):
        ds_train = ds_train.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    else:
        ds_train = ds_train.shuffle(
            ds_info.splits['train'].num_examples // 10, reshuffle_each_iteration=True)

    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info
