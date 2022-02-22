import gin
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


@gin.configurable
def preprocess(image, label, img_height, img_width):
    """
    Dataset preprocessing: Normalizing and resizing
    NOTE: Normalize each image into range [0,1]  (this reduces the effect of brighter/darker image)

    """

    # Normalize image
    img_max = tf.reduce_max(image)
    image = tf.cast(image, tf.float32) / tf.cast(img_max, tf.float32)

    # Resize image -> no need since done in tfrecords
    # image = tf.image.resize(image, size=(img_height, img_width))

    return image, label


rng = tf.random.Generator.from_seed(123)


@gin.configurable
def random_apply(image, func, p=0.25):
    if tf.random.uniform([]) < p:
        image = func(image)
    else:
        image = image
    return image


def random_brightness(image):
    return tf.image.stateless_random_brightness(image, 0.1, seed=rng.make_seeds(2)[0])


def random_contrast(image):
    return tf.image.stateless_random_contrast(image, 0.45, 1, seed=rng.make_seeds(2)[0])


def random_flip_left_right(image):
    return tf.image.stateless_random_flip_left_right(image, seed=rng.make_seeds(2)[0])


def random_flip_up_down(image):
    return tf.image.stateless_random_flip_up_down(image, seed=rng.make_seeds(2)[0])

def random_crop(image):
    
    width = gin.query_parameter('preprocess.img_width')
    height = gin.query_parameter('preprocess.img_height')
    
    crop_width = int(width/1.2)
    crop_height = int(height/1.2)

    image = tf.image.stateless_random_crop(image, (crop_width, crop_height, 3), seed=rng.make_seeds(2)[0])
    image = tf.image.resize(image, [width, height])
    
    return image


def random_rotate(image):
    # rotate randomly between +90° and -90°
    angle = tf.random.uniform(
        shape=[], minval=-np.pi/2, maxval=np.pi/2, dtype=tf.dtypes.float32,
        seed=123)

    return tfa.image.rotate(image, angle)


def augment(image, label):
    """Data augmentation"""
    image = random_apply(image, random_brightness)
    image = random_apply(image, random_contrast)
    image = random_apply(image, random_flip_left_right)
    image = random_apply(image, random_flip_up_down)
    image = random_apply(image, random_rotate)
    # image = random_apply(image, random_crop)
    image = tf.clip_by_value(image, 0, 1)
    return image, label
