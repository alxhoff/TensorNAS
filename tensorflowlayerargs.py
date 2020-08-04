from enum import Enum, auto

class Conv2DArgs(Enum):
  'Args needed for creating Conv2DArgs layer, list not complete'
  FILTERS = auto()
  KERNEL_SIZE = auto()
  STRIDES = auto()
  INPUT_SIZE = auto()


class MaxPool2DArgs(Enum):
  'Args needed for creating MaxPool2D layer, list not complete'
  POOL_SIZE = auto()
  STRIDES = auto()


class ReshapeArgs(Enum):
  'Args needed for creating Reshape layer'
  TARGET_SHAPE = auto()


class DenseArgs(Enum):
  'Args needed for creating Dense layer, list not complete'
  UNITS = auto()
  ACTIVATION = auto()


class DropoutArgs(Enum):
  'Args needed for creating Dropout layer, list not complete'
  RATE = auto()
