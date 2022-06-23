from TensorNAS.Core import EnumWithNone
from enum import auto
from TensorNAS.Core.Layer import Layer
from TensorNAS.Core.LayerMutations import MutateNumGroups


def get_divisors(n):
    for i in range(1, int(n / 2) + 1):
        if n % i == 0:
            yield i
    yield n


def shuffle_channels(input_tensor, num_groups):
    import tensorflow as tf

    n, h, w, c = input_tensor.shape.as_list()
    grouped_channels = c // num_groups
    reshaped_input = tf.reshape(input_tensor, [-1, h, w, num_groups, grouped_channels])
    transformed_input = tf.transpose(reshaped_input, [0, 1, 2, 4, 3])
    output = tf.reshape(transformed_input, [-1, h, w, c])
    return output


class Args(EnumWithNone):

    NUM_GROUPS = auto()


class Layer(Layer, MutateNumGroups):
    def _gen_args(self, input_shape, args):
        from random import choice

        groups = choice(list(get_divisors(input_shape[-1])))

        if args:
            if self.get_args_enum().NUM_GROUPS in args:
                groups = args.get(self.get_args_enum().NUM_GROUPS)

        return {self.get_args_enum().NUM_GROUPS: groups}

    def get_output_shape(self):
        return self.inputshape.get()

    def get_keras_layers(self, input_tensor):
        return shuffle_channels(
            input_tensor, self.args.get(self.get_args_enum().NUM_GROUPS)
        )
