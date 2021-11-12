from TensorNAS.Layers.Conv2D import Layer

"""The number of depthwise convolution output channels for each input channel. The total number of depthwise convolution 
output channels will be equal to filters_in * depth_multiplier."""
"""
Given a 4D input tensor ('NHWC' or 'NCHW' data formats) and a filter
 tensor of shape [filter_height, filter_width, in_channels, channel_multiplier] 
 containing input_channels convolutional filters of depth 1, depthwise_conv2d applies a different filter
  to each input channel (expanding from 1 channel to channel_multiplier channels for each), then concatenates 
  the results together. The output has in_channels * channel_multiplier channels.
  
  If we want output channels= output/ratio*(ratio-1), then we would have to calculate the 
"""


class Layer(Layer):
    def get_keras_layers(self, input_tensor):
        import tensorflow as tf

        return tf.keras.layers.DepthwiseConv2D(
            kernel_size=self.args.get(self.get_args_enum().KERNEL_SIZE),
            strides=self.args.get(self.get_args_enum().STRIDES),
            padding=self.args.get(self.get_args_enum().PADDING).value(),
            dilation_rate=self.args.get(self.get_args_enum().DILATION_RATE),
            depth_multiplier=1,
            activation=self.args.get(self.get_args_enum().ACTIVATION).value(),
        )(input_tensor)
