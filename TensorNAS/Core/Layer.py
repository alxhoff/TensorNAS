import random
import re
from abc import ABC, abstractmethod
from enum import Enum


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


class Layer(ABC):
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

    def __init__(self, input_shape, parent_block, args=None):

        self.args_enum = self._get_args_enum()
        if args:
            if isinstance(args, list):
                args = dict(args)
                new_dict = {}

                for key, val in args.items():
                    if isinstance(key, str):
                        new_dict[
                            [i for i in self.args_enum if i.name == key][0]
                        ] = args[key]
                args = new_dict

        self.parent_block = parent_block
        self.args = self._gen_args(input_shape, args)
        self.inputshape = LayerShape()
        self.outputshape = LayerShape()
        self.mutation_funcs = [
            func
            for func in dir(self)
            if callable(getattr(self, func)) and re.search(r"^_mutate", func)
        ]

        self.inputshape.set(input_shape)
        self.outputshape.set(self.get_output_shape())

    @classmethod
    def _get_module(cls):
        return cls.__module__

    @classmethod
    def _get_m_name(cls):
        ret = re.findall(r"^(.*)\.([a-zA-Z0-9]*$)", cls._get_module())
        if len(ret):
            return ret[0]
        else:
            return None

    @classmethod
    def _get_parent_module(cls):
        ret = cls._get_m_name()
        if ret:
            if len(ret) >= 2:
                return ret[0]

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

    @classmethod
    def _get_args_enum(cls):
        from importlib import import_module

        try:
            args = import_module(cls._get_module()).Args
            return args
        except Exception:
            try:
                args = import_module(cls._get_parent_module()).Args
                return args
            except Exception as e:
                raise (
                    "{} doesn't have args enum.Enum 'Args' implemented".format(
                        cls.get_name()
                    )
                )

    def set_input_shape(self, input_shape):
        self.inputshape.set(input_shape)

    def set_output_shape(self, output_shape):
        self.outputshape.set(output_shape)

    def get_sb_count(self):
        return 0

    def get_args_enum(self):
        return self.args_enum

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

    def mutate(self, verbose=False):
        if self.mutation_funcs:
            mutate_eval = "self." + random.choice(self.mutation_funcs)
            if verbose:
                print("[MUTATE] invoking `{}`".format(mutate_eval))
            eval(mutate_eval)()

    @abstractmethod
    def _gen_args(self, input_shape, args):
        return NotImplementedError

    @abstractmethod
    def get_output_shape(self):
        return NotImplementedError

    @abstractmethod
    def get_keras_layers(self, input_tensor):
        return NotImplementedError

    def _args_to_JSON(self):

        args = dict(self.args)

        ret = []

        for arg in args:
            ret += [[arg.name, value] for key, value in args.items() if arg == key]

        return ret

    def toJSON(self):

        json_dict = {
            "input_shape": self.inputshape.get(),
            "output_shape": self.outputshape.get(),
            "mutation_funcs": self.mutation_funcs,
            "args": self._args_to_JSON(),
        }

        return json_dict

    def get_ascii_tree(self):
        return str(self)


class ArgActivations(str, Enum):
    ELU = "elu"
    EXPONENTIAL = "exponential"
    HARD_SIGMOID = "hard_sigmoid"
    LINEAR = "linear"
    RELU = "relu"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    SOFTPLUS = "softplus"
    TANH = "tanh"


class ArgPadding(str, Enum):
    VALID = "valid"
    SAME = "same"


class ArgRegularizers(str, Enum):
    L1 = "L1"
    L1L2 = "L1L2"
    L2 = "L2"


def gen_2d_kernel_size(input_size):
    kernel_size = random.choice(range(1, input_size, 2))
    return (kernel_size, kernel_size)


def gen_3d_strides(max_bound):
    stride_size = random.randint(1, max_bound)
    return (stride_size, stride_size, stride_size)


def gen_2d_strides(max_bound):
    stride_size = random.randint(1, max_bound)
    return (stride_size, stride_size)


def gen_1d_strides(max_bound):
    return random.randint(1, max_bound)


def gen_3d_poolsize(max_bound):
    size = random.randint(1, max_bound)
    return (size, size, size)


def gen_2d_poolsize(max_bound):
    size = random.randint(1, max_bound)
    return (size, size)


def gen_1d_poolsize(max_bound):
    return random.randint(1, max_bound)


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


def gen_activation():
    return random.choice(list(ArgActivations))


def gen_groups(max_bound):
    return random.randint(1, max_bound)
