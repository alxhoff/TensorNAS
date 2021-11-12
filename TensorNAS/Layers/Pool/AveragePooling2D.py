from TensorNAS.Layers.Pool.MaxPool2D import Layer


class Layer(Layer):
    def get_keras_layers(self, input_tensor):
        import tensorflow as tf

        return tf.keras.layers.AveragePooling2D(
            input_shape=self.inputshape.get(),
            pool_size=self.args.get(self.get_args_enum().POOL_SIZE),
            strides=self.args.get(self.get_args_enum().STRIDES),
            padding=self.args.get(self.get_args_enum().PADDING).value(),
        )(input_tensor)
