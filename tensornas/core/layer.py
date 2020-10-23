import importlib
import random
import re
from abc import ABC, abstractmethod


class LayerShape:
    def __init__(self, dimensions=None):
        self.dimensions = dimensions

    def __str__(self):
        if self.dimensions:
            return str(self.dimensions)
        else:
            return "?"

    def __eq__(self, other):
        if isinstance(other, LayerShape):
            return self.dimensions == other.dimensions
        return False

    def set(self, dimensions):
        self.dimensions = dimensions

    def get(self):
        return tuple(self.dimensions)


class NetworkLayer(ABC):
    """
    Layers are implemented using an abstract class that must provide a number of abstract methods. This is done such
    that the implemented layers can be loaded in a plugin fashion from the layers sub-package. This allows for users
    to provide implementations for the layers they require without having to manually integrate them into the rest of
    tensornas.

    Layers are to be created in the layers sub-package and be named according to Keras convention. The layer itself
    should be implemented as a class called Layer.

    It is possible to add another layer of depth when creating layer types, eg. Dense layers can be created as hidden
    dense layers of output dense layers. When doing so the sub-class layers should be placed in another sub-package,
    the sub-package being named according to Keras' layer naming. If the layers share the arguments enum Args then it
    can also be placed inside the __init__.py of the created sub-package. See the Dense sub-package for an example.

    Mutating:
    Mutation happens by randomly calling a mutation function implemented within the class, ideally each implemented
    function should mutate one property of the layer that the class represents. Calling of these mutation functions is
    done through naming convention, each mutation function should be prefixed with '_mutate'.

    Args:
    Each network layer needs an enum.Enum where all of the possible arguments are listed. It must have the name Args
    and be in the same module as the Layer class. If the layer is a sub-class of a Keras type layed, eg. a hidden Dense
    layer then the Args enum can be placed inside the parent sub-package such that it can be shared between the
    sub-classed layers.
    """

    def __init__(self, input_shape, args=None):
        self.args_enum = self._get_args_enum()
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
        self.validate(repair=True)

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
        try:
            args = importlib.import_module(cls._get_module()).Args
            return args
        except Exception:
            try:
                args = importlib.import_module(cls._get_parent_module()).Args
                return args
            except Exception as e:
                raise (
                    "{} doesn't have args enum.Enum 'Args' implemented".format(
                        cls.get_name()
                    )
                )

    def get_args_enum(self):
        return self.args_enum

    def print(self):
        print(
            "Layer:{} {}-> {}".format(
                self.get_name(), self.inputshape, self.outputshape
            )
        )
        try:
            arg_list = list(self.get_args_enum())
            for param, param_value in self.args.items():
                if isinstance(param, int):
                    name = arg_list[param - 1].name
                else:
                    name = param.name
                print("{}: {}".format(name, param_value))
        except Exception:
            pass
        print("")

    def repair(self):
        pass

    def validate(self, repair=True):
        return True

    def mutate(self, verbose=False):
        if self.mutation_funcs:
            mutate_eval = "self." + random.choice(self.mutation_funcs)
            if verbose:
                print("[MUTATE] invoking `{}`".format(mutate_eval))
            eval(mutate_eval)()

    @classmethod
    @abstractmethod
    def _gen_args(cls, input_shape, args):
        return NotImplementedError

    @abstractmethod
    def get_output_shape(self):
        return NotImplementedError

    @abstractmethod
    def get_keras_layer(self):
        return NotImplementedError
