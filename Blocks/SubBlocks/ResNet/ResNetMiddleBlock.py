from TensorNAS.Core.Block import Block
from enum import Enum, auto
from TensorNAS.Core import EnumWithNone


class Args(EnumWithNone):

    FILTERS = auto()


class Block(Block):

    MAX_SUB_BLOCKS = 5

    class SubBlocks(Enum):

        CONV2D = auto()
        BATCH_NORMALIZATION = auto()
        ACTIVATION = auto()

    def generate_constrained_middle_sub_blocks(self, input_shape, args=None):
        from TensorNAS.Layers.Conv2D.Conv2D import Layer as Conv2D
        from TensorNAS.Layers.BatchNormalization import Layer as BatchNormalization
        from TensorNAS.Layers.Activation import Layer as Activation

        from TensorNAS.Layers.Conv2D import Args as conv2d_args
        from TensorNAS.Layers.Activation import Args as activation_args

        from TensorNAS.Core.Layer import (
            ArgPadding,
            ArgRegularizers,
            ArgActivations,
            ArgInitializers,
        )

        filters = 16
        if args:
            if self.get_args_enum().FILTERS in args:
                filters = args.get(self.get_args_enum().FILTERS)

        layers = []
        layers.append(
            Conv2D(
                input_shape=input_shape,
                parent_block=self,
                args={
                    conv2d_args.FILTERS: filters,
                    conv2d_args.KERNEL_SIZE: (3, 3),
                    conv2d_args.STRIDES: (1, 1),
                    conv2d_args.PADDING: ArgPadding.SAME,
                    conv2d_args.REGULARIZER: (ArgRegularizers.L2, 1e-4),
                    conv2d_args.ACTIVATION: ArgActivations.NONE,
                    conv2d_args.INITIALIZER: ArgInitializers.HE_NORMAL,
                },
            )
        )
        layers.append(
            BatchNormalization(
                input_shape=layers[-1].get_output_shape(), parent_block=self
            )
        )
        layers.append(
            Activation(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                args={activation_args.ACTIVATION: ArgActivations.RELU},
            )
        )
        layers.append(
            Conv2D(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                args={
                    conv2d_args.FILTERS: filters,
                    conv2d_args.KERNEL_SIZE: (3, 3),
                    conv2d_args.STRIDES: (1, 1),
                    conv2d_args.PADDING: ArgPadding.SAME,
                    conv2d_args.REGULARIZER: (ArgRegularizers.L2, 1e-4),
                    conv2d_args.ACTIVATION: ArgActivations.NONE,
                    conv2d_args.INITIALIZER: ArgInitializers.HE_NORMAL,
                },
            )
        )
        layers.append(
            BatchNormalization(
                input_shape=layers[-1].get_output_shape(), parent_block=self
            )
        )

        return layers

    def generate_sub_block(self, input_shape, layer_type):
        from TensorNAS.Layers.Activation import Layer as Activation

        if layer_type == self.SubBlocks.CONV2D:
            from TensorNAS.Layers.Conv2D.Conv2D import Layer as Conv2D
            from TensorNAS.Layers.Conv2D import Args as conv2d_args
            from TensorNAS.Core.Layer import (
                ArgPadding,
                ArgRegularizers,
                ArgActivations,
                ArgInitializers,
            )

            return [
                Conv2D(
                    input_shape=input_shape,
                    parent_block=self,
                    args={
                        conv2d_args.FILTERS: 16,
                        conv2d_args.KERNEL_SIZE: (3, 3),
                        conv2d_args.STRIDES: (1, 1),
                        conv2d_args.PADDING: ArgPadding.SAME,
                        conv2d_args.REGULARIZER: (ArgRegularizers.L2, 1e-4),
                        conv2d_args.ACTIVATION: ArgActivations.NONE,
                        conv2d_args.INITIALIZER: ArgInitializers.HE_NORMAL,
                    },
                )
            ]
        elif layer_type == self.SubBlocks.BATCH_NORMALIZATION:
            from TensorNAS.Layers.BatchNormalization import Layer as BatchNormalization

            return [BatchNormalization(input_shape=input_shape, parent_block=self)]
        elif layer_type == self.SubBlocks.ACTIVATION:
            from TensorNAS.Layers.Activation import Args as activation_args
            from TensorNAS.Core.Layer import ArgActivations

            return [
                Activation(
                    input_shape=input_shape,
                    parent_block=self,
                    args={activation_args.ACTIVATION: ArgActivations.RELU},
                )
            ]

        return []

    def get_keras_layers(self, input_tensor):
        """
        We need to override the get_keras_layer function as the default function simply builds the model
        sequentially and does not account for parallel/skip connections such as those used in ResNet.
        """
        from TensorNAS.Tools.TensorFlow import shortcut
        from TensorNAS.Layers.Activation import Layer as Activation
        from TensorNAS.Layers.Activation import Args as activation_args
        from TensorNAS.Core.Layer import ArgActivations

        tmp = input_tensor
        for sb in self.input_blocks + self.middle_blocks + self.output_blocks:
            tmp = sb.get_keras_layers(tmp)
        tmp = shortcut(input_tensor, tmp)

        act_layer = Activation(
            input_shape=tuple(tmp.shape[1:]),
            parent_block=None,
            args={activation_args.ACTIVATION: ArgActivations.RELU},
        )

        ret = act_layer.get_keras_layers(tmp)

        return ret
