def shortcut(input, residual):
    import tensorflow as tf

    input_shape = tf.keras.backend.int_shape(input)
    residual_shape = tf.keras.backend.int_shape(residual)
    equal_channels = input_shape[3] == residual_shape[3]

    shortcut = input

    if (input_shape[1] % residual_shape[1] or input_shape[2] % residual_shape[2]) and (
        input_shape[1] != residual_shape[1] and input_shape[2] != residual_shape[2]
    ):
        kernel_size = (
            input_shape[1] - residual_shape[1] + 1,
            input_shape[2] - residual_shape[2] + 1,
        )
        stride_width = 1
        stride_height = 1
    else:
        kernel_size = (1, 1)
        stride_width = int(round(input_shape[1] / residual_shape[1]))
        stride_height = int(round(input_shape[2] / residual_shape[2]))

    if not equal_channels:
        shortcut = tf.keras.layers.Conv2D(
            filters=residual_shape[3],
            kernel_size=kernel_size,
            strides=(stride_width, stride_height),
            padding="valid",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
        )(input)

    ret = tf.keras.layers.add([shortcut, residual])
    return ret
