from enum import Enum, auto

import TensorNAS.Core.LayerArgs as la
from TensorNAS.Core.Layer import Layer
from TensorNAS.Core.Util import dimension_mag, mutate_int


def get_divisors(n):
    for i in range(1, int(n / 2) + 1):
        if n % i == 0:
            yield i
    yield n


def shuffle_channels(input_tensor, num_groups):
    import tensorflow as tf

    print("Num groups: {}".format(num_groups))
    n, h, w, c = input_tensor.shape.as_list()
    print("Shuffle input: {}".format(input_tensor.shape.as_list()))
    grouped_channels = c // num_groups
    reshaped_input = tf.reshape(input_tensor, [-1, h, w, num_groups, grouped_channels])
    print("First reshape: {}".format(reshaped_input.shape.as_list()))
    transformed_input = tf.transpose(reshaped_input, [0, 1, 2, 4, 3])
    print("Transposed: {}".format(reshaped_input.shape.as_list()))
    output = tf.reshape(transformed_input, [-1, h, w, c])
    print("Output reshape: {}".format(output.shape.as_list()))
    return output


class Args(Enum):
    "Args needed for creating Shuffle layer"
    NUM_GROUPS = auto()


class Layer(Layer):
    def _gen_args(self, input_shape, args):
        from random import choice

        groups = choice(list(get_divisors(input_shape[-1])))

        if args:
            if self.get_args_enum().NUM_GROUPS in args:
                groups = args.get(self.get_args_enum().NUM_GROUPS)

        return {self.get_args_enum().NUM_GROUPS: groups}

    def _mutate_num_groups(self):
        self.args[self.get_args_enum().NUM_GROUPS] = mutate_int(1, self.MAX_NUM_GROUPS)

    def get_output_shape(self):
        return self.inputshape.get()

    def get_keras_layers(self, input_tensor):
        return shuffle_channels(
            input_tensor, self.args.get(self.get_args_enum().NUM_GROUPS)
        )
