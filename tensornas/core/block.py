import itertools
import random
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
        - Mutate
        - Generate constrained output sub blocks
        - Generate constrained input sub blocks
        - Generate random sub block

    Optional methods:
        - Validate
        - Get keras model
        - Print self
        - Output shape
        - Check new layer type

    Should not be overridden:
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

    @abstractmethod
    def mutate(self):
        """Mutates the block, this can be done by changing direct child blocks or by calling into the child blocks to
        mutate them"""
        return NotImplemented

    @abstractmethod
    def generate_constrained_output_sub_blocks(self, input_shape):
        """This method is called after the sub-blocks have been generated to generate the required blocks,
        creating blocks at the end of the sub-block sequence. An example of this would be the placement
        of a classification layer at the end of a model.
        """
        pass

    @abstractmethod
    def generate_constrained_input_sub_blocks(self, input_shape):
        """This method is called before the sub-blocks have been generated to generate the required blocks,
        creating blocks at the beginning of the sub-block sequence. An example of this would be the placement
        of a convolution layer at the beginning of a model.
        """
        pass

    @abstractmethod
    def generate_random_sub_block(self, input_shape, layer_type):
        """This method appends a randomly selected possible sub-block to the classes sub-block list, The block type is
        passed in as layer_type which is randomly selected from the provided enum SUB_BLOCK_TYPES which stores the
        possible sub block types. This function is responsible for instantiating each of these sub blocks if required.
        """
        return NotImplementedError

    def check_new_layer_type(self, layer_type):
        """
        This function is called when the next layer type is randomly generated, inside the function the user can perform
        checks to check if the next layer type is allowed, more likely will be checks to see if the next layer type is
        invalid, eg. two sequential flatten layers.
        """
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
                out_shape = self.__get_cur_output_shape()
                while True:
                    block = self.generate_random_sub_block(
                        out_shape,
                        self.__get_random_sub_block_type(),
                    )
                    if block:
                        self.sub_blocks.append(block)
                        break

    def get_output_shape(self):
        """
        Returns the output shape of the block
        """
        return self.sub_blocks[-1].get_output_shape()

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
                if self.check_new_layer_type(next_type):
                    return next_type
        else:
            return None

    def get_iterator(self):
        return itertools.chain(*[block.sub_blocks for block in self.sub_blocks])

    def get_keras_layers(self):
        """By default this method simply calls this method in all child blocks, it should be overriden for layer
        blocks, ie. blocks that are leaves within the block hierarchy and contain a keras layer, such blocks should
        return an appropriately instantiated keras layer object"""
        ret = []
        for sb in self.sub_blocks:
            ret.append(sb.get_keras_layers())

        if hasattr(ret[0], "__iter__"):
            return list(itertools.chain(*ret))
        else:
            return ret

    def __get_cur_output_shape(self):
        if len(self.sub_blocks):
            ret = self.sub_blocks[-1].get_output_shape()
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
        for sb in self.sub_blocks:
            sb.print()

    def print_self(self):
        pass

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

        while True:
            self.sub_blocks = []
            if self.MAX_SUB_BLOCKS:
                self.generate_constrained_input_sub_blocks(input_shape)
                self.__generate_sub_blocks()
                self.generate_constrained_output_sub_blocks(
                    self.__get_cur_output_shape()
                )
                if self.validate():
                    return
            else:
                return
