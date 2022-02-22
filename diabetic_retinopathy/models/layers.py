import gin
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from keras.layers.merge import concatenate
from tensorflow.keras import layers, Sequential


@gin.configurable
def vgg_block(inputs, filters, kernel_size):
    """A single VGG block consisting of two convolutional layers, followed by a max-pooling layer.

    Parameters:
        inputs (Tensor): input of the VGG block
        filters (int): number of filters used for the convolutional layers
        kernel_size (tuple: 2): kernel size used for the convolutional layers, e.g. (3, 3)

    Returns:
        (Tensor): output of the VGG block
    """

    out = tf.keras.layers.Conv2D(
        filters, kernel_size, padding='same', activation=tf.nn.relu)(inputs)
    out = tf.keras.layers.Conv2D(
        filters, kernel_size, padding='same', activation=tf.nn.relu)(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    return out


@gin.configurable
def cnn_block(inputs, filters, kernel_size):
    """A CNN block of ous always consits of a Convolutional layer """

    output = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size,
                                    activation="relu", kernel_regularizer=regularizers.L1(l1=0.01, ))(inputs)

    output = tf.keras.layers.MaxPool2D((2,2))(output)
    output = tf.keras.layers.BatchNormalization()(output)
    
    return output



@gin.configurable
def inception_module(inputs, f1=64, f2_in=96, f2_out=128, f3_in=16, f3_out=32, f4_out=32, normalise=True):
    # 1x1 conv
    conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(inputs)
    # 3x3 conv
    conv3_1 = Conv2D(f2_in, (1,1), padding='same', activation='relu')(inputs)
    conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3_1)
    # 5x5 conv
    conv5_1 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(inputs)
    conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5_1)
    # 3x3 max pooling
    pool = MaxPool2D((3,3), strides=(1,1), padding='same')(inputs)
    pool = Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)
    # concatenate filters
    out = concatenate([conv1, conv3, conv5, pool], axis=-1)
 
    # if normalise:
    #     out = BatchNormalization(out)
    return out


@gin.configurable
def res_block(inputs, filters, kernel_size, strides):
    out = layers.Conv2D(filters, kernel_size, padding="same", strides=strides,
                        activation="relu", kernel_regularizer=regularizers.l2(0.01))(inputs)
    out = layers.BatchNormalization()(out)
    out = layers.Conv2D(filters, kernel_size, padding="same", strides=1,
                        kernel_regularizer=regularizers.l2(0.01))(out)
    out = layers.BatchNormalization()(out)

    if strides != 1:
        identity = layers.Conv2D(filters, kernel_size=1, padding="same", strides=strides)(inputs)
    else:
        identity = layers.Conv2D(filters, kernel_size=1, padding="same", strides=1)(inputs)

    out = layers.add([out, identity])
    out = layers.ReLU()(out)
    return out


def res_blocks(inputs, filters_num, block_num, strides=1):
    out = res_block(inputs, filters=filters_num, strides=strides)
    for i in range(2, block_num+1):
        out = res_block(out, filters=filters_num, strides=1)
    return out


def stem(input_shape):
    return Sequential(
        [
            tf.keras.Input(shape=input_shape),
            layers.Conv2D(32, 3, 1, padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(2, 2), strides=1, padding="same"),
        ], name="stem"
    )
