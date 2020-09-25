from abc import ABC, abstractmethod
import itertools
import random
from tensornasmutator import mutate_enum_i


class Block(ABC):
    """
    An abstract class that all model blocks are derived from. Thus all model blocks, regardless of their depth within
    the model architecture binary tree they must implement all of the abstract methods defined within this class.
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
    def validate(self):
        """This function will check if the generated block sequence is valid

        @return True if the sub-block sequence is valid else False
        """
        return NotImplementedError

    @abstractmethod
    def generate_constrained_sub_blocks(self):
        """This method is called after the sub-blocks have been generated to generate the required blocks, this is
        generally a specific type of block at the beginning or end of the sub-block sequence. An example of this
        would be the placement of a classification layer at the end of a model
        """
        return NotImplementedError

    def __generate_sub_blocks(self):
        """Subclasses of Block should not populate their sub-block lists but instead implement this function which
        will handle this
        """
        for i in range(random.randrange(1, self.MAX_SUB_BLOCKS - 1)):
            self.sub_blocks.append(self.gen_block_from_type_enum())

    @abstractmethod
    def get_random_sub_block(self, layer_type):
        """This method appends a randomly selected possible sub-block to the classes sub-block list, done using
        __get_random_sub_block_type
        """
        return NotImplementedError

    @classmethod
    def get_random_sub_block_type(cls):
        """This method returns a random enum value of the block's possible sub-blocks"""
        return mutate_enum_i(cls.SUB_BLOCK_TYPES)

    def gen_block_from_type_enum(self):
        type = self.get_random_sub_block_type()
        return self.get_random_sub_block(type)

    def get_iterator(self):
        return itertools.chain(block.sub_blocks for block in self.sub_blocks)

    def __init__(self, input_shape):
        self.input_shape = input_shape

        # Loop until valid sub-block sequence is created
        while True:
            self.sub_blocks = []
            self.__generate_sub_blocks()
            self.generate_constrained_sub_blocks()
            if self.validate():
                break
