import gin
import sys
import logging
import tensorflow as tf
import tensorflow_datasets as tfds

from input_pipeline.preprocessing import preprocess


def parse_and_decode_record(record):
    """Parse and decode in one function call to simply .map() it when used"""
    record = parse_record(record)
    record = decode_record(record)
    return record

def parse_record(record):
    name_to_features = {
        'features': tf.io.FixedLenFeature([], tf.string),
        'labels': tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(record, name_to_features)

def decode_record(record):
    features = tf.io.parse_tensor(record['features'], out_type=tf.double)
    labels = tf.io.parse_tensor(record['labels'], out_type=tf.double)
    return (features, labels)


@gin.configurable
def load(name, data_dir, window_length, window_shift, batch_size, fc=False):
    """
    Load the dataset from tf-record files 

    Args:
        name (str): name of the datasat that should be loaded (by now only hapt is implemented)
        data_dir (str): path to tf-records directory
        window_length (int, optional): defaults to 250.
        batch_size (int): batch size the loaded data should have.
        fc (int, optional): cutoff frequency to laod. Defaults to False (no execution).
    
    Returns:
        [tf-dataset]: returns train, test and val dataset ready to be used by tensorflow
    """
    
    # expand data_dir name with window_length and size
    if fc > 0 and fc < 25:
        data_dir = data_dir + f"/wl{str(window_length)}_ws{str(window_shift)}_fc{str(fc)}"
    else:
        data_dir = data_dir + f"/wl{str(window_length)}_ws{str(window_shift)}"
    
    if name == "hapt":
        logging.info(f"Preparing dataset {name}...")
        raw_train_ds = tf.data.TFRecordDataset(data_dir + "/train.tfrecords")
        raw_val_ds = tf.data.TFRecordDataset(data_dir + "/validation.tfrecords")
        raw_test_ds = tf.data.TFRecordDataset(data_dir + "/test.tfrecords")
        
        # decode raw data
        decoded_ds_train = raw_train_ds.map(parse_and_decode_record)
        decoded_ds_val = raw_val_ds.map(parse_and_decode_record)
        decoded_ds_test = raw_test_ds.map(parse_and_decode_record)
        

        return prepare_ds(decoded_ds_train, decoded_ds_val, decoded_ds_test, batch_size=batch_size)

    else:
        logging.error('Currently only hapt dataset is implemented. Exiting now!')
        sys.exit(0)


@gin.configurable
def prepare_ds(ds_train, ds_val, ds_test, batch_size, caching):
    
    # Prepare training dataset

    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(64)
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

    return ds_train, ds_val, ds_test
