from TensorNAS.Core.Block import Block


class ParallelMidBlock(Block):
    def get_keras_layers(self, input_tensor):
        import tensorflow as tf

        tmp = input_tensor

        for sb in self.input_blocks:
            tmp = sb.get_keras_layers(tmp)

        mid_output_tensors = []

        for sb in self.middle_blocks:
            mid_output_tensors.append(sb.get_keras_layers(tmp))

        tmp = tf.keras.layers.concatenate([tmp] + mid_output_tensors)

        for sb in self.output_blocks:
            tmp = sb.get_keras_layers(tmp)

        return tmp
