from TensorNAS.Layers.Conv2D import Layer


class Layer(Layer):
    def get_keras_layer(self, input_tensor):
        import tensorflow as tf

        kernel_regularizer = None
        if self.get_args_enum().KERNEL_REGULARIZER in self.args:
            r_args = self.args.get(self.get_args_enum().KERNEL_REGULARIZER)
            kernel_regularizer = eval("tf.keras.regularizers.{}".format(r_args[0]))(
                *list(r_args)
            )

        try:
            return tf.keras.layers.Conv2D(
                filters=self.args.get(self.get_args_enum().FILTERS),
                kernel_size=self.args.get(self.get_args_enum().KERNEL_SIZE),
                strides=self.args.get(self.get_args_enum().STRIDES),
                input_shape=self.inputshape.get(),
                activation=self.args.get(self.get_args_enum().ACTIVATION).value,
                padding=self.args.get(self.get_args_enum().PADDING).value,
                dilation_rate=self.args.get(self.get_args_enum().DILATION_RATE),
                kernel_regularizer=kernel_regularizer,
            )(input_tensor)
        except Exception as e:
            raise e
