from TensorNAS.Layers.Conv2D import Layer


class Layer(Layer):
    def get_keras_layers(self, input_tensor):
        import tensorflow as tf

        return tf.keras.layers.SeparableConv2D(
            filters=self.args.get(self.get_args_enum().FILTERS),
            kernel_size=self.args.get(self.get_args_enum().KERNEL_SIZE),
            strides=self.args.get(self.get_args_enum().STRIDES),
            padding=self.args.get(self.get_args_enum().PADDING).value,
            dilation_rate=self.args.get(self.get_args_enum().DILATION_RATE),
            depth_multiplier=1,
            activation=self.args.get(self.get_args_enum().ACTIVATION).value,
        )(input_tensor)
