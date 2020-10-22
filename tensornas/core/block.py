import itertools
import random
import re
from abc import ABC, abstractmethod

from tensornas.core.util import mutate_enum_i


class Block(ABC):
    """
    An abstract class that all model blocks are derived from. Thus all model blocks, regardless of their depth within
    the model architecture binary tree they must implement all of the abstract methods defined within this class.

    Required properties:
        - SUB_BLOCK_TYPES
        - MAX_SUB_BLOCKS

    Required (abstract) methods:
        - Generate random sub block

    Optional methods:
        - Generate constrained output sub blocks
        - Generate constrained input sub blocks
        - Mutate self
        - Validate
        - Get keras model
        - Print self
        - Output shape
        - Check new layer type

    Should not be overridden:
        - Mutate
        - __init__
    """

    @property
    @classmethod
    @abstractmethod
    def SUB_BLOCK_TYPES(cls):
        """The enum type storing all the possible sub-blocks is required to be passed to the child class as it is
        used during random selection of sub-block blocks
        """
        return NotImplementedError

    @property
    @classmethod
    @abstractmethod
    def MAX_SUB_BLOCKS(cls):
        """Constraining attribute of each Block sub-class that must be set"""
        return NotImplementedError

    def _mutate_self(self, verbose=False):
        """
        An optional function that allows for the block to mutate itself during mutation
        """
        return False

    def _mutate_subblock(self, verbose=False):
        if len(self.middle_blocks):
            choice_index = random.choice(range(len(self.middle_blocks)))
            if verbose:
                print("[MUTATE] middle block #{}".format(choice_index))
            self.middle_blocks[choice_index].mutate(verbose=verbose)

    def mutate(self, self_mutate_rate=0.0, verbose=False):
        """Similar to NetworkLayer objects, block mutation is a randomized call to any methods prexied with `_mutate`,
        this includes the defaul `_mutate_subblock`.

        The implementation of a block should as such then present the possible mutation possibilities as a collection
        of `_mutate` functions. Generally mutation will call the default `_mutate_subblock` method to invoke mutation
        in a randomly selected sub-block.

        If specific mutation operations are thus required they can be implemented. Among the default mutation functions
        is the `_mutate_self` function which directly mutates the block instead of calling mutate in a sub-block.
        The function by default does nothing and returns False, in such a case another mutate function is called.
        If one wishes to implement `_mutate_self` then it should return True to stop the subsequent
        re-invoking of mutate.

        The probability of mutating the block itself instead of it's sub-block is pass in as self_mutate_rate."""
        if random.random() < self_mutate_rate:
            if self._mutate_self(verbose=verbose):
                return
        if self.mutation_funcs:
            mutate_eval = "self." + random.choice(self.mutation_funcs)
            if verbose:
                print("[MUTATE] invoking `{}`".format(mutate_eval))
            eval(mutate_eval)(verbose=verbose)

    def generate_constrained_output_sub_blocks(self, input_shape):
        """This method is called after the sub-blocks have been generated to generate the required blocks which are
        appended to the output_blocks list. An example of this would be the placement
        of a classification layer at the end of a model.
        """
        pass

    def generate_constrained_input_sub_blocks(self, input_shape):
        """This method is called before the sub-blocks have been generated to generate the required blocks which are
        appended to the input_blocks list. An example of this would be the placement
        of a convolution layer at the beginning of a model.
        """
        pass

    @abstractmethod
    def generate_random_sub_block(self, input_shape, layer_type):
        """This method appends a randomly selected possible sub-block to the classes middle_blocks list, The block type is
        passed in as layer_type which is randomly selected from the provided enum SUB_BLOCK_TYPES which stores the
        possible sub block types. This function is responsible for instantiating each of these sub blocks if required.
        """
        return NotImplementedError

    def check_next_layer_type(self, prev_layer_type, next_layer_type):
        """
        This function is called when the next layer type is randomly generated, inside the function the user can perform
        checks to check if the next layer type is allowed, more likely will be checks to see if the next layer type is
        invalid, eg. two sequential flatten layers.

        It should be noted that the previous layer is given as a SupportedLayer, this is an enum value generated by the
        tensornas.layers package by scanning the provided modules in the package. The next layer's type is given in
        terms of the possible sub-block types for the current block. As such you must make an apples to oranges
        comparison.
        """
        return True

    def _check_layer_types(self, next_layer_type):
        """
        An intermediate function that checks if there is a previous layer to be passed to the check_next_layer_type
        function.
        """
        if len(self.middle_blocks):
            return self.check_next_layer_type(
                self.middle_blocks[-1].layer_type, next_layer_type
            )
        return True

    def validate(self):
        """This function will check if the generated block sequence is valid. Default implementation can be used which
        always returns true, ie. the block is always considered valid.

        @return True if the sub-block sequence is valid else False
        """
        return True

    def __generate_sub_blocks(self):
        """Subclasses of Block should not populate their sub-block lists but instead implement this function which
        will handle this. Generated blocks that are not valid
        """
        if self.MAX_SUB_BLOCKS:
            for i in range(random.randrange(1, self.MAX_SUB_BLOCKS)):
                out_shape = self._get_cur_output_shape()
                while True:
                    block = self.generate_random_sub_block(
                        out_shape,
                        self.__get_random_sub_block_type(),
                    )
                    if block:
                        self.middle_blocks.append(block)
                        break

    def get_output_shape(self):
        """
        Returns the output shape of the block
        """
        return self.middle_blocks[-1].get_output_shape()

    def get_input_shape(self):
        """
        Returns in the input shape of the block
        """
        return self.input_shape

    def __get_random_sub_block_type(self):
        """This method returns a random enum value of the block's possible sub-blocks"""
        if self.SUB_BLOCK_TYPES:
            while True:
                next_type = mutate_enum_i(self.SUB_BLOCK_TYPES)
                if self._check_layer_types(next_type):
                    return next_type
        else:
            return None

    def get_keras_layers(self):
        """By default this method simply calls this method in all child blocks, it should be overriden for layer
        blocks, ie. blocks that are leaves within the block hierarchy and contain a keras layer, such blocks should
        return an appropriately instantiated keras layer object"""
        ret = []
        for sb in self.input_blocks + self.middle_blocks + self.output_blocks:
            ret.append(sb.get_keras_layers())

        if hasattr(ret[0], "__iter__"):
            return list(itertools.chain(*ret))
        else:
            return ret

    def _get_cur_output_shape(self):
        if len(self.output_blocks):
            ret = self.output_blocks[-1].get_output_shape()
        elif len(self.middle_blocks):
            ret = self.middle_blocks[-1].get_output_shape()
        elif len(self.input_blocks):
            ret = self.input_blocks[-1].get_output_shape()
        else:
            ret = self.get_input_shape()
        try:
            assert ret, "Input shape None"
            return ret
        except Exception as e:
            exit(e)

    def print(self):
        """
        Function useful for print debugging, the function by default invokes print self and then invokes printing in all
        child nodes. If you wish to print all children nodes then only override print_self and not print_self
        """
        self.print_self()
        for sb in self.input_blocks + self.middle_blocks + self.output_blocks:
            sb.print()

    def print_self(self):
        pass

    def print_tree(self):
        """"""

    def display_tree(self):
        from tensornas.core.util import block_width, stack_str_blocks

        if not len(self.input_blocks + self.middle_blocks + self.output_blocks):
            return str(self.layer_type).split(".")[-1]

        child_strs = [
            child.display_tree()
            for child in self.input_blocks + self.middle_blocks + self.output_blocks
        ]
        child_widths = [block_width(s) for s in child_strs]

        # How wide is this block?
        display_width = max(
            len(str(self.layer_type).split(".")[-1]),
            sum(child_widths) + len(child_widths) - 1,
        )

        # Determines midpoints of child blocks
        child_midpoints = []
        child_end = 0
        for width in child_widths:
            child_midpoints.append(child_end + (width // 2))
            child_end += width + 1

        # Builds up the brace, using the child midpoints
        brace_builder = []
        for i in range(display_width):
            if i < child_midpoints[0] or i > child_midpoints[-1]:
                brace_builder.append(" ")
            elif i in child_midpoints:
                brace_builder.append("+")
            else:
                brace_builder.append("-")
        brace = "".join(brace_builder)

        name_str = "{:^{}}".format(str(self.layer_type).split(".")[-1], display_width)
        below = stack_str_blocks(child_strs)

        return name_str + "\n" + brace + "\n" + below

    def __init__(self, input_shape, parent_block, layer_type):
        """
        The init sequence of the Block class should always be called at the end of a subclasse's __init__, via
        super().__init__ if a subclass is to implement its own __init__ method.

        This can be required if the block needs to take in additional parameters when being created, eg. a classification
        block needs to known the number of output classes that it must classify. Thus such an implementation will
        read in its required init arguments and then invoke the Block.__init__ such that the defined Block init
        sequence is performed.

        The layer type is used by the parent block to identify the block's type from it's enum of valid sub-block types.
        """
        self.input_shape = input_shape
        self.parent_block = parent_block
        self.layer_type = layer_type
        self.mutation_funcs = [
            func
            for func in dir(self)
            if callable(getattr(self, func)) and re.search(r"^_mutate(?!_self)", func)
        ]

        while True:
            self.input_blocks = []
            self.middle_blocks = []
            self.output_blocks = []
            if self.MAX_SUB_BLOCKS:
                self.generate_constrained_input_sub_blocks(input_shape)
                self.__generate_sub_blocks()
                self.generate_constrained_output_sub_blocks(
                    self._get_cur_output_shape()
                )
                if self.validate():
                    return
            else:
                return
