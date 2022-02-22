import gin
from tcn import TCN
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

@gin.configurable
def model_tcn(window_size, nb_filters, kernel_size, nb_stacks, dropout_rate, kernel_initializer):
    """

    Args:
        window_size: window size of the data 
        nb_filters: Integer. The number of filters to use in the convolutional layers.
        kernel_size: Integer. The size of the kernel to use in each convolutional layer.
        nb_stacks:Integer. The number of stacks of residual blocks to use.
        dropout_rate: 0.05 in origin method
        kernel_initializer: default "he_normal"

    Returns: (None, nb_filters)

    """
    input_shape = (window_size, 6)
    tcn_layer = TCN(input_shape=input_shape,
                    nb_filters=nb_filters,
                    kernel_size=kernel_size,
                    nb_stacks=nb_stacks,
                    dropout_rate=dropout_rate,
                    kernel_initializer=kernel_initializer
                    )
    model = Sequential([
        tcn_layer,
        Dense(12, activation="softmax")
    ])
    return model
