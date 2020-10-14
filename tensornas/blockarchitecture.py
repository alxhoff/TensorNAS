from tensornas.block import Block


class BlockArchitecture(Block):
    """
    A block architectures, eg. a classification architecture is one that provides a specified
    number of probability outputs that are used in the classification of some input.

    The abstract block architecture class defines the methods that must be implemented to allow for a type of block
    architecture to be created, namely what sort of sub-blocks the block architecture can generate.
    """

    def mutate(self):
        pass

    def generate_random_sub_block(self, layer_type):
        pass

    def generate_constrained_input_sub_blocks(self, input_shape):
        pass

    def generate_constrained_output_sub_blocks(self, input_shape):
        pass

    def get_output_shape(self):
        pass
