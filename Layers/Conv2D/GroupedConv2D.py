from TensorNAS.Layers.Conv2D import Layer


class Layer(Layer):
    def get_keras_layers(self, input_tensor):
        import tensorflow as tf

        return tf.keras.layers.Conv2D(
            filters=self.args.get(self.get_args_enum().FILTERS),
            kernel_size=self.args.get(self.get_args_enum().KERNEL_SIZE),
            strides=self.args.get(self.get_args_enum().STRIDES),
            input_shape=self.inputshape.get(),
            activation=self.args.get(self.get_args_enum().ACTIVATION).value(),
            padding=self.args.get(self.get_args_enum().PADDING).value(),
            dilation_rate=self.args.get(self.get_args_enum().DILATION_RATE),
            groups=self.args.get(self.get_args_enum().GROUPS),
        )(input_tensor)
