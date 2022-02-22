import gin
import tensorflow as tf
import logging

import tensorflow.keras as keras
from models.layers import vgg_block, cnn_block, inception_module, res_blocks, stem
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras import regularizers


@gin.configurable
def vgg_like(input_shape, n_classes, base_filters, n_blocks, dense_units, dropout_rate):
    """Defines a VGG-like architecture.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        base_filters (int): number of base filters, which are doubled for every VGG block
        n_blocks (int): number of VGG blocks
        dense_units (int): number of dense units
        dropout_rate (float): dropout rate

    Returns:
        (keras.Model): keras model object
    """

    assert n_blocks > 0, 'Number of blocks has to be at least 1.'

    inputs = tf.keras.Input(input_shape)
    out = vgg_block(inputs, base_filters)
    for i in range(1, n_blocks):
        out = vgg_block(out, base_filters * 2 ** (i))
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg_like')


@gin.configurable
def team20_cnn_01(input_shape, kernel_size, strides, filters, max_pool_dimension, dropout_rate):

    model = tf.keras.Sequential(name="team20_cnn_01")

    model.add(tf.keras.Input(shape=input_shape,))

    model.add(Conv2D(
        filters=filters[0], kernel_size=kernel_size[0],
        strides=strides, activation="relu",
        kernel_regularizer=regularizers.L1(l1=0.01, )))

    model.add(MaxPool2D(pool_size=max_pool_dimension))
    model.add(BatchNormalization())

    model.add(Conv2D(
        filters=filters[1], kernel_size=kernel_size[1],
        strides=strides, activation="relu",
        kernel_regularizer=regularizers.L1(l1=0.01, )))

    model.add(MaxPool2D(pool_size=max_pool_dimension))
    model.add(BatchNormalization())

    model.add(Conv2D(
        filters=filters[2], kernel_size=kernel_size[2],
        strides=strides, activation="relu",
        kernel_regularizer=regularizers.L1(l1=0.01, )))

    model.add(tf.keras.layers.GlobalAveragePooling2D())

    model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(
        units=16, kernel_regularizer=regularizers.l2(0.001), activation="relu"))

    model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(
        units=2, kernel_regularizer=regularizers.l2(0.001)))

    model.build()

    logging.info(f"team20_cnn_01 input shape:  {model.input_shape}")
    logging.info(f"team20_cnn_01 output shape: {model.output_shape}")

    return model


@gin.configurable
def team20_cnn_02(input_shape, kernel_size, strides, filters, max_pool_dimension, dropout_rate):

    model = tf.keras.Sequential(name="team20_cnn_02")

    model.add(tf.keras.Input(shape=input_shape,))

    model.add(Conv2D(
        filters=filters[0], kernel_size=kernel_size[0],
        strides=strides, activation="relu",
        kernel_regularizer=regularizers.L1(l1=0.01, )))

    model.add(Conv2D(
        filters=filters[1], kernel_size=kernel_size[0],
        strides=strides, activation="relu",
        kernel_regularizer=regularizers.L1(l1=0.01, )))

    model.add(MaxPool2D(pool_size=max_pool_dimension))
    model.add(BatchNormalization())

    model.add(Conv2D(
        filters=filters[2], kernel_size=kernel_size[1],
        strides=strides, activation="relu",
        kernel_regularizer=regularizers.L1(l1=0.01, )))

    model.add(MaxPool2D(pool_size=max_pool_dimension))
    model.add(BatchNormalization())

    model.add(Conv2D(
        filters=filters[3], kernel_size=kernel_size[3], strides=strides, activation="relu",
        kernel_regularizer=regularizers.L1(l1=0.01, )))

    model.add(MaxPool2D(pool_size=max_pool_dimension))
    model.add(BatchNormalization())

    model.add(Conv2D(
        filters=filters[4], kernel_size=kernel_size[4], strides=strides, activation="relu",
        kernel_regularizer=regularizers.L1(l1=0.01, )))

    model.add(tf.keras.layers.GlobalAveragePooling2D())

    model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(
        units=16, kernel_regularizer=regularizers.l2(0.001)))

    model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(
        units=2, kernel_regularizer=regularizers.l2(0.001)))

    model.build()

    logging.info(f"team20_cnn_02 input shape:  {model.input_shape}")
    logging.info(f"team20_cnn_02 output shape: {model.output_shape}")

    return model


@gin.configurable
def cnn_blueprint(input_shape, n_classes, base_filters, n_blocks_pool, n_blocks_normal,
                  dense_units, dropout_rate, strides, strides_pre):
    """CNN model with modular size and blocks defined by TEAM20.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        base_filters (int): number of base filters, which are doubled every block
        dense_units (int): number of dense units
        dropout_rate (float): dropout rate

    Returns:
        (keras.Model): keras model object
    """

    assert n_blocks_normal > 0, 'Number of blocks has to be at least 1.'

    assert n_blocks_pool > 0

    inputs = tf.keras.Input(input_shape)

    out = Conv2D(
        filters=base_filters, kernel_size=3, strides=strides_pre, activation="relu", padding="same",
        kernel_regularizer=regularizers.L1(l1=0.01, ))(inputs)

    for i in range(2, n_blocks_normal+1):
        out = Conv2D(
            filters=base_filters * 2 ** (i-1), kernel_size=3, strides=strides, activation="relu",
            kernel_regularizer=regularizers.L1(l1=0.01, ))(inputs)

    for i in range(1, n_blocks_pool+1):
        out = cnn_block(out, base_filters * 2 ** (i+n_blocks_normal-1))
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = Dense(dense_units, activation=tf.nn.relu)(out)
    out = Dropout(dropout_rate)(out)
    outputs = Dense(n_classes)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='cnn')


@gin.configurable
def inception_blueprint(input_shape, n_classes, base_filters, n_blocks_incep, n_block_CNN, dense_units, dropout_rate):
    """ Second CNN-blueprint of group 20 using inception module and normal convolutional layers """

    inputs = tf.keras.Input(input_shape)

    out = inception_module(inputs)

    for i in range(2, n_blocks_incep+1):
        out = inception_module(out)

    for i in range(1, n_block_CNN+1):
        out = cnn_block(out, base_filters * 2 ** (i))
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='inception')


@gin.configurable
def resnet(input_shape, n_classes, fc_units, filters_num, dropout_rate, layer_dim):
    inputs = tf.keras.Input(shape=input_shape),
    out = stem(input_shape)(inputs)
    out = res_blocks(out, filters_num, block_num=layer_dim[0], strides=2)
    out = res_blocks(out, filters_num * 2, block_num=layer_dim[1], strides=2)
    out = layers.Conv2D(32, 3, padding="same")(out)
    out = layers.GlobalAvgPool2D()(out)
    out = layers.Dense(fc_units, activation="relu",
                       kernel_regularizer=regularizers.l2(0.01))(out)
    out = layers.Dropout(dropout_rate)(out)
    outputs = layers.Dense(n_classes)(out)
    return tf.keras.Model(inputs, outputs, name="resnet")


@gin.configurable
def transfer_model(dropout_rate, dense_units, layer_num, basemodel_name='EfficientNetB0'):
    if basemodel_name == 'EfficientNetB0':
        basemodel = keras.applications.EfficientNetB0(
            include_top=False, input_shape=(256, 256, 3))

    if basemodel_name == 'InceptionResNet':
        basemodel = keras.applications.InceptionResNetV2(
            include_top=False, input_shape=(256, 256, 3))

    if basemodel_name == 'Xception':
        basemodel = keras.applications.Xception(
            include_top=False, input_shape=(256, 256, 3))

    basemodel.trainable = False
    for layer in basemodel.layers[-layer_num:]:
        if not isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = True
    model = keras.Sequential(
        [
            keras.Input(shape=(256, 256, 3)),
            # keras.layers.Rescaling(scale=255.0),
            basemodel,
            layers.Conv2D(32, 3, padding="same"),
            keras.layers.GlobalAvgPool2D(),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(
                dense_units, kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.01)),
            keras.layers.Dense(2)
        ]
    )
    return model
