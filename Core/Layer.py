import random
import re
from abc import ABC, abstractmethod
from TensorNAS.Core import EnumWithNone


class LayerShape:
    def __init__(self, dimensions=None):
        self.dimensions = dimensions

    def __str__(self):
        if self.dimensions:
            return str(self.dimensions)
        else:
            raise Exception("Unable to get layer's shape")

    def __eq__(self, other):
        if isinstance(other, LayerShape):
            return self.dimensions == other.dimensions
        return False

    def set(self, dimensions):
        self.dimensions = dimensions

    def get(self):
        return tuple(self.dimensions)


from TensorNAS.Core.Block import BaseBlock


class Layer(BaseBlock):
    """
    Layers are implemented using an abstract class that must provide a number of abstract methods. This is done such
    that the implemented Layers can be loaded in a plugin fashion from the Layers sub-package. This allows for users
    to provide implementations for the Layers they require without having to manually integrate them into the rest of
    TensorNAS Framework.

    Layers are to be created in the Layers sub-package and be named according to Keras convention. The layer itself
    should be implemented as a class called Layer.

    It is possible to add another layer of depth when creating layer types, eg. Dense Layers can be created as hidden
    dense Layers of output dense Layers. When doing so the sub-class Layers should be placed in another sub-package,
    the sub-package being named according to Keras' layer naming. If the Layers share the arguments enum Args then it
    can also be placed inside the __init__.py of the created sub-package. See the Dense sub-package for an example.

    Mutating:
    Mutation happens by randomly calling a mutation function implemented within the class, ideally each implemented
    function should mutate one property of the layer that the class represents. Calling of these mutation functions is
    done through naming convention, each mutation function should be prefixed with '_mutate'.

    Args:
    Each network layer needs an enum.Enum where all of the possible arguments are listed. It must have the name Args
    and be in the same module as the Layer class. If the layer is a sub-class of a Keras type layed, eg. a hidden Dense
    layer then the Args enum can be placed inside the parent sub-package such that it can be shared between the
    sub-classed Layers.
    """

    class SubBlocks:
        """
        Required by parent class Block
        """

    def __init__(self, input_shape, parent_block, args=None):

        super().__init__(input_shape=input_shape, parent_block=parent_block, args=args)

        self.args = self._gen_args(input_shape, args)
        self.inputshape = LayerShape()
        self.outputshape = LayerShape()

        self.inputshape.set(input_shape)
        self.outputshape.set(self.get_output_shape())

    @classmethod
    def get_parent_name(cls):
        ret = re.findall(r".*\.([a-zA-Z0-9]*$)", cls._get_parent_module())
        if len(ret):
            return ret[0]
        else:
            return None

    @classmethod
    def get_name(cls):
        ret = cls._get_m_name()
        if ret:
            if len(ret) >= 2:
                return ret[1]

    def set_input_shape(self, input_shape):
        self.inputshape.set(input_shape)

    def set_output_shape(self, output_shape):
        self.outputshape.set(output_shape)

    def get_sb_count(self):
        return 0

    def __str__(self):
        ret = "Layer:{} {}-> {}, ".format(
            self.get_name(), self.inputshape, self.outputshape
        )
        try:
            arg_list = list(self.get_args_enum())
            for param, param_value in self.args.items():
                if isinstance(param, int):
                    name = arg_list[param - 1].name
                else:
                    name = param.name
                ret += "{}: {}, ".format(name, param_value)
            ret += "\n"
        except Exception:
            pass
        return ret

    def print(self):
        print(str(self))

    def mutate(self, verbose=False, **kwargs):
        return self._invoke_random_mutation_function(verbose=verbose, **kwargs)

    @abstractmethod
    def _gen_args(self, input_shape, args):
        return NotImplementedError

    @abstractmethod
    def get_output_shape(self):
        return NotImplementedError

    @abstractmethod
    def get_keras_layers(self, input_tensor):
        return NotImplementedError

    def get_ascii_tree(self):
        return str(self)


class ArgActivations(EnumWithNone):
    NONE = None
    ELU = "elu"
    EXPONENTIAL = "exponential"
    HARD_SIGMOID = "hard_sigmoid"
    LINEAR = "linear"
    RELU = "relu"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    SOFTPLUS = "softplus"
    TANH = "tanh"


class ArgPadding(EnumWithNone):
    VALID = "valid"
    SAME = "same"


class ArgRegularizers(EnumWithNone):
    NONE = None
    L1 = "L1"
    L1L2 = "L1L2"
    L2 = "L2"


# Given the large number, only those used were added
class ArgInitializers(EnumWithNone):
    HE_NORMAL = "he_normal"
    HE_UNIFORM = "he_uniform"
    GLOROT_UNIFORM = "glorot_uniform"


def gen_2d_kernel_size(input_size):
    kernel_size = int(random.choice(range(1, input_size, 2)))
    return (kernel_size, kernel_size)


def gen_3d_strides(max_bound):
    stride_size = int(random.randint(1, max_bound))
    return (stride_size, stride_size, stride_size)


def gen_2d_strides(max_bound):
    stride_size = int(random.randint(1, max_bound))
    return (stride_size, stride_size)


def gen_1d_strides(max_bound):
    return int(random.randint(1, max_bound))


def gen_3d_poolsize(max_bound):
    size = int(random.randint(1, max_bound))
    return (size, size, size)


def gen_2d_poolsize(max_bound):
    size = int(random.randint(1, max_bound))
    return (size, size)


def gen_1d_poolsize(max_bound):
    return int(random.randint(1, max_bound))


def gen_2d_dilation():
    # TODO
    return (1, 1)


def gen_dropout(max):
    while True:
        ret = round(random.uniform(0, max), 2)
        if ret != 0.0:
            return ret


def gen_padding():
    return random.choice(list(ArgPadding))


# Desired regularizer needs to be provided as a tuple containing the regularizer's name as a list of it's init arguments
def gen_regularizer(value=None):
    from TensorNAS.Core.Layer import ArgRegularizers

    if value:
        if value[0] != ArgRegularizers.NONE:
            import tensorflow as tf

            value = eval("tf.keras.regularizers.{}".format(value[0]))(
                *(value[1] if isinstance(value[1], list) else [value[1]])
            )

    return None


def gen_initializer(value="glorot_uniform"):
    return value


def gen_activation():
    return random.choice(list(ArgActivations))


def gen_groups(max_bound):
    return random.randint(1, max_bound)
