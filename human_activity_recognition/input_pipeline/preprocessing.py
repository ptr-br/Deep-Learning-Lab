import tensorflow as tf

def preprocess(features, labels):
    """Dataset preprocessing"""
    
    # cast to int
    labels = tf.cast(labels, tf.int32)
    
    # subtract to get labels from 0 to 11 for sparse categorical cross entropy loss
    labels = tf.subtract(labels, 1)

    return features, labels

def augment(features, labels):
    # TODO: Maybe ook if any reasonable data augmentation is possible ...
    return features, features