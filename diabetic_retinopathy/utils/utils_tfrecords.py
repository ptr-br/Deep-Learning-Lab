import tensorflow as tf

# Encoding
# Convert values to compatible tf.Example types. - Adapted from tensorflow docs


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Decoding

def parse_and_decode_record(record):
    """Parse and decode in one function call"""
    record = parse_record(record)
    record = decode_record(record)
    return record


def parse_record(record):
    name_to_features = {
        'label': tf.io.FixedLenFeature([1], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(record, name_to_features)


def decode_record(record):
    image = tf.io.decode_jpeg(record['image_raw'], channels=3)
    label = record['label']
    return (image, label)
