from TensorNAS.Core.Block import Block
from enum import Enum, auto


class Block(Block):

    MAX_SUB_BLOCKS = 3

    class SubBlocks(Enum):

        CONV2D = auto()

    def generate_constrained_input_sub_blocks(self, input_shape):
        import random
        from TensorNAS.Layers.Conv2D.PointwiseConv2D import Layer as PointwiseConv2D
        from TensorNAS.Layers.Pool.SameMaxPool2D import Layer as SameMaxPool2D

        layers = []
        if random.randint(0, 1):
            layers = [
                SameMaxPool2D(
                    input_shape=input_shape,
                    parent_block=self,
                )
            ]
        if len(layers):
            layers.append(
                PointwiseConv2D(
                    input_shape=layers[0].get_output_shape(),
                    parent_block=self,
                )
            )
        else:
            layers.append(
                PointwiseConv2D(
                    input_shape=input_shape,
                    parent_block=self,
                )
            )
        return layers

    def generate_random_sub_block(self, input_shape, layer_type):
        from TensorNAS.Layers.Conv2D.SameConv2D import Layer as SameConv2D

        if layer_type == self.SubBlocks.CONV2D:
            return [
                SameConv2D(
                    input_shape=input_shape,
                    parent_block=self,
                )
            ]
        return []

    def get_keras_layers(self, input_tensor):
        tmp = input_tensor
        for sb in self.input_blocks + self.middle_blocks + self.output_blocks:
            tmp = sb.get_keras_layers(tmp)
        return tmp
