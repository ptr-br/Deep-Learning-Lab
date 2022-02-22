import gin
import keras
import logging
from tensorflow.keras import layers, Input, Model


def get_layer(rnn_type, rnn_units, return_sequence, droput, kernel_initializer, last):

    if rnn_type.lower() == "lstm":
        rnn_layer_last = layers.LSTM(units=rnn_units, return_sequences=return_sequence,
                                     dropout=droput, kernel_initializer=kernel_initializer)
        rnn_layer_stack = layers.LSTM(units=rnn_units, return_sequences=True,
                                      dropout=droput, kernel_initializer=kernel_initializer)

    elif rnn_type.lower() == "gru":
        rnn_layer_last = layers.GRU(units=rnn_units, return_sequences=return_sequence,
                                    dropout=droput, kernel_initializer=kernel_initializer)
        rnn_layer_stack = layers.GRU(units=rnn_units, return_sequences=True,
                                     dropout=droput, kernel_initializer=kernel_initializer)

    elif rnn_type.lower() == "rnn":
        rnn_layer_last = layers.SimpleRNN(units=rnn_units, return_sequences=return_sequence,
                                          dropout=droput, kernel_initializer=kernel_initializer)
        rnn_layer_stack = layers.SimpleRNN(units=rnn_units, return_sequences=True,
                                           dropout=droput, kernel_initializer=kernel_initializer)
    if last:
        return rnn_layer_last
    else:
        return rnn_layer_stack


@gin.configurable
def rnn(
    n_classes,
    window_length,
    rnn_units,
    rnn_type,
    num_rnn,
    dense_units,
    num_dense,
    dropout_dense,
    dropout_rnn,
    bi_direction,
    kernel_initializer,
    return_sequence=False,
):
    """
    Scalable architecture for different recurrent layers  

    Args:
        n_classes (int): number of ouput classes
        window_length (int): length of the sliding window
        rnn_units (int): number of rnn units
        rnn_type (str): type of the recurrent layers (lstm | gru | rnn )
        num_rnn (int): number of recurrent layers
        dense_units (int): number of dense units
        num_dense (int): number of dense layers
        dropout_rate (float): dropout rate between dense layers
        bi_direction (bool): state if bidirectional should be used
        return_sequence (bool): whether to return the last output. in the output sequence, or the full sequence
        kernel_initializer (str): default "glorot_uniform", can also use "he_normal", "lecun_normal"

    Returns:
        keras.model: model object
    """

    model = keras.Sequential([keras.Input(shape=(window_length, 6)),])

    for _ in range(num_rnn - 1):
        layer = get_layer(rnn_type, rnn_units, return_sequence, dropout_rnn, kernel_initializer, last=False)
        if bi_direction:
            layer = layers.Bidirectional(layer)
        model.add(layer)
        if layer.output.shape[1] > n_classes:
            model.add(layers.MaxPool1D(2))
        model.add(layers.BatchNormalization())
    layer = get_layer(rnn_type, rnn_units, return_sequence, dropout_rnn, kernel_initializer, last=True)
    model.add(layer)

    for elm in range(1, num_dense + 1):
        if dense_units >= 2 * n_classes and elm > 1:
            dense_units = int(dense_units / (1.75))
        model.add(layers.Dense(dense_units, kernel_regularizer=keras.regularizers.L1L2(l1=0.01, l2=0.01)))
        model.add(layers.Dropout(dropout_dense))

    model.add(layers.Dense(n_classes, activation="softmax"))

    logging.info(f"rnn input shape:  {model.input_shape}")
    logging.info(f"rnn output shape: {model.output_shape}")

    return model


## Deprecated
# class RNN(keras.Model):
#     def __init__(self, n_classes, rnn_units, rnn_type, num_rnn, dense_units,
#                  num_dense, return_sequence, dropout_rate,bi_direction):
#         super(RNN,self).__init__()
#         self.n_classes = n_classes
#         self.rnn_units = rnn_units
#         self.rnn_type = rnn_type
#         self.num_rnn = num_rnn
#         self.dense_units = dense_units
#         self.num_dense = num_dense
#         self.return_sequence = return_sequence
#         self.dropout_rate = dropout_rate
#         self.bi_direction = bi_direction
#         self.rnn_layer = self.build_rnn_layer()
#
#
#     def build_rnn_layer(self):
#         if self.rnn_type.lower() == "lstm":
#             self.rnn_layer = layers.LSTM(units=self.rnn_units, return_sequences=self.return_sequence)
#
#         elif self.rnn_type.lower() == "gru":
#             self.rnn_layer = layers.GRU(units=self.rnn_units, return_sequences=self.return_sequence)
#
#         elif self.rnn_type.lower() == "rnn":
#             self.rnn_layer = layers.SimpleRNN(units=self.rnn_units, return_sequences=self.return_sequence)
#
#         if self.bi_direction:
#             self.rnn_layer = layers.Bidirectional(self.rnn_layer)
#
#         return self.rnn_layer
#
#
#     def call(self, input):
#         out = layers.Dropout(self.dropout_rate)(input)
#         for _ in range(self.num_rnn):
#             out = self.rnn_layer(out)
#         for _ in range(self.num_dense):
#             out = layers.Dense(self.dense_units, kernel_regularizer=keras.regularizers.L1L2(l1=0.01, l2=0.01))(out)
#             out = layers.Dropout(self.dropout_rate)(out)
#         output = layers.Dense(self.n_classes)
#         return output
#
#
#
