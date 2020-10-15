from enum import Enum, auto

from tensornas.core.networklayer import NetworkLayer
from tensornas.core.util import MutationOperators, mutate_tuple


class Args(Enum):
    "Args needed for creating MaxPool2D layer, list not complete"
    POOL_SIZE = auto()
    STRIDES = auto()
    PADDING = auto()


class Layer(NetworkLayer):
    MAX_POOL_SIZE = 7
    MAX_STRIDE = 7

    def _mutate_pool_size(self, operator=MutationOperators.SYNC_STEP):
        self.args[self.get_args_enum().POOL_SIZE.value] = mutate_tuple(
            self._pool_size(), 1, self.MAX_POOL_SIZE, operator=operator
        )

    def _mutate_strides(self, operator=MutationOperators.SYNC_STEP):
        self.args[self.get_args_enum().STRIDES.value] = mutate_tuple(
            self._strides(), 1, self.MAX_STRIDE, operator=operator
        )
