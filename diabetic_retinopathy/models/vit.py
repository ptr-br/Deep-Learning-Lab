import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

input_shape = (256, 256, 3)
patch_size = 16
patch_stride = 16
num_patches = (input_shape[0] // patch_size) ** 2  # Number of total patches = 256
embedding_dim = 64
mlp_dim = 64
num_heads = 4
attention_dropout_rate = 0.2
num_transformer_blocks = 2


class PatchExtractEmbedding(layers.Layer):
    def __init__(self, patch_size, patch_stride, embed_dim, num_patches, **kwargs):
        super(PatchExtractEmbedding, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.embed_layer = layers.Embedding(input_dim=self.num_patches, output_dim=self.embed_dim)
        self.proj_layer = layers.Dense(self.embed_dim)

    def call(self, images):
        batch_size = tf.shape(images)[0]
        # patches.shape = (128, 16, 16, 16x16x3)
        patches = tf.image.extract_patches(
            images=images,
            sizes=(1, self.patch_size, self.patch_size, 1),
            strides=(1, self.patch_stride, self.patch_stride, 1),
            rates=(1, 1, 1, 1),
            padding="VALID",
        )
        patch_dim = patches.shape[-1]
        patches = tf.reshape(patches, (batch_size, self.num_patches, patch_dim))  # (128, 256, 768)

        total_patches = patches.shape[1]
        position = tf.range(0, total_patches)
        # (256, 64)
        position_embed = self.embed_layer(position)
        projection = self.proj_layer(patches)  # (128, 256, 64)

        return position_embed + projection


def mlp(embedding_dim, mlp_dim):
    return keras.Sequential(
        [
            layers.Dense(mlp_dim),
            layers.Dropout(0.2),
            layers.Dense(embedding_dim),
            layers.Dropout(0.2)
        ])


def transformer_block(x, num_heads, embedding_dim, attention_dropout_rate):
    residual_1 = x
    x = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embedding_dim, dropout=attention_dropout_rate
    )(x, x)
    x = layers.add([x, residual_1])
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    residual_2 = x
    x = mlp(embedding_dim, mlp_dim)(x)
    x = layers.add([x, residual_2])
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    return x


def vit():
    inputs = keras.Input(shape=input_shape)
    out = PatchExtractEmbedding(patch_size, patch_stride, embedding_dim, num_patches)(inputs)
    for _ in range(num_transformer_blocks):
        out = transformer_block(out, num_heads, embedding_dim, attention_dropout_rate)
    out = layers.GlobalAvgPool1D()(out)
    out = layers.Dense(5)(out)
    outputs = layers.Dense(1, activation="sigmoid")(out)
    return keras.Model(inputs=inputs, outputs=outputs)



