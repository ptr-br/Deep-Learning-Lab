import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, Sequential


def res_block(inputs, filters, kernel_size=3, strides=1):
    out = layers.Conv2D(filters, kernel_size, padding="same", strides=strides, activation="relu")(inputs)
    out = layers.BatchNormalization()(out)
    out = layers.Conv2D(filters, kernel_size, padding="same", strides=1)(out)
    out = layers.BatchNormalization()(out)

    if strides != 1:
        identity = layers.Conv2D(filters, kernel_size=1, padding="same", strides=strides)(inputs)
    else:
        identity = layers.Conv2D(filters, kernel_size=1, padding="same", strides=1)(inputs)

    out = layers.add([out, identity])
    out = layers.ReLU()(out)
    return out


def build_blocks(inputs, filters_num, block_num, strides=1):
    out = res_block(inputs, filters=filters_num, strides=strides)
    for i in range(2, block_num+1):
        out = res_block(out, filters=filters_num, strides=1)
    return out


def stem(input_shape):
    return Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, 3, 1, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(2, 2), strides=1, padding="same"),
        ], name="stem"
    )


def resnet(input_shape=(256, 256, 3), n_classes=2, fc_units=256, dropout_rate=0.2, layer_dim=[2, 2, 2, 2]):
    inputs = keras.Input(shape=input_shape),
    out = stem(input_shape)(inputs)
    out = build_blocks(out, filters_num=64, block_num=layer_dim[0])
    out = build_blocks(out, filters_num=64, block_num=layer_dim[1])
    out = build_blocks(out, filters_num=128, block_num=layer_dim[2])
    out = build_blocks(out, filters_num=128, block_num=layer_dim[3])
    out = layers.GlobalAvgPool2D()(out)
    out = layers.Dense(fc_units, activation="relu")(out)
    out = layers.Dropout(dropout_rate)(out)
    outputs = layers.Dense(n_classes)(out)
    return keras.Model(inputs, outputs, name="resnet")


images = tf.random.normal((10, 256, 256, 3))
resnet_18 = resnet(layer_dim=[2, 2, 2, 2])
print(resnet_18(images))
resnet_18.summary()





