import tensorflow as tf
from tensornas.layers.MaxPool import Layer


class Layer(Layer):
    def get_keras_layer(self, input_tensor):
        return tf.keras.layers.GlobalAveragePooling2D(data_format=None)(input_tensor)
