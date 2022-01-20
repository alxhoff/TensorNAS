from TensorNAS.Layers.Pool import Layer


class Layer(Layer):
    def get_output_shape(self):
        inp = self.inputshape.get()
        return inp[-1]

    def get_keras_layers(self, input_tensor):
        import tensorflow as tf

        return tf.keras.layers.GlobalAveragePooling2D(data_format="channels_last")(
            input_tensor
        )
