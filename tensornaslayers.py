from tensorflowlayerargs import *
from tensornasmutator import *
import keras


class LayerShape:
    def __init__(self, dimensions=None):
        self.dimensions = dimensions

    def __str__(self):
        if self.dimensions:
            return str(self.dimensions)
        else:
            return "?"

    def set(self, dimensions):
        self.dimensions = dimensions


class ModelLayer:
    "Common layer properties"

    def __init__(self, name, args=None):
        self.name = name
        if args:
            self.args = args
        else:
            self.args = {}

        self.inputshape = LayerShape()
        self.outputshape = LayerShape()

    def getname(self):
        return self.name

    def getargs(self):
        return self.args

    def print(self):
        print("{} [{}]-> [{}]".format(self.name, self.inputshape, self.outputshape))
        for param_name, param_value in self.args.items():
            print("{}: {}".format(param_name, param_value))


class Conv2DLayer(ModelLayer):
    MUTATABLE_PARAMETERS = 6
    MAX_FILTER_COUNT = 128
    MAX_KERNEL_DIMENSION = 7
    MAX_STRIDE = 7
    MAX_DILATION = 5

    def __init__(
        self,
        filters,
        kernel_size,
        strides,
        input_size,
        padding=PaddingArgs.SAME.value,
        dilation_rate=(0, 0),
        activation=Activations.RELU.value,
    ):
        super().__init__("Conv2D")

        self.args[Conv2DArgs.FILTERS.name] = filters
        self.args[Conv2DArgs.KERNEL_SIZE.name] = kernel_size
        self.args[Conv2DArgs.STRIDES.name] = strides
        self.args[Conv2DArgs.INPUT_SIZE.name] = input_size
        self.args[Conv2DArgs.PADDING.name] = padding
        self.args[Conv2DArgs.DILATION_RATE.name] = dilation_rate
        self.args[Conv2DArgs.ACTIVATION.name] = activation

        self.inputshape.set(input_size)
        self.calc_output_shape()

    def _filters(self):
        return self.args[Conv2DArgs.FILTERS.name]

    def _kernel_size(self):
        return self.args[Conv2DArgs.KERNEL_SIZE.name]

    def _strides(self):
        return self.args[Conv2DArgs.STRIDES.name]

    def _input_size(self):
        return self.args[Conv2DArgs.INPUT_SIZE.name]

    def _padding(self):
        return self.args[Conv2DArgs.PADDING.name]

    def _dilation_rate(self):
        return self.args[Conv2DArgs.DILATION_RATE.name]

    def _activation(self):
        return self.args[Conv2DArgs.ACTIVATION.name]

    def _single_stride(self):
        st = self._strides()
        if st[0] == 1 and st[1] == 1:
            return True
        return False

    def _single_dilation_rate(self):
        dr = self._dilation_rate()
        if dr[0] == 1 and dr[1]:
            return True
        return False

    def _mutate_filters(self, operator=MutationOperators.STEP):
        self.args[Conv2DArgs.FILTERS.name] = mutate_int(
            self._filters(), 1, Conv2DLayer.MAX_FILTER_COUNT, operator
        )

    def _mutate_kernel_size(self, operator=MutationOperators.SYNC_STEP):
        self.args[Conv2DArgs.KERNEL_SIZE.name] = mutate_tuple(
            self._kernel_size(), 1, Conv2DLayer.MAX_KERNEL_DIMENSION, operator
        )

    def _mutate_strides(self, operator=MutationOperators.SYNC_STEP):
        self.args[Conv2DArgs.STRIDES.name] = mutate_tuple(
            self._strides(), 1, Conv2DLayer.MAX_STRIDE, operator
        )

    def _mutate_padding(self):
        self.args[Conv2DArgs.PADDING.name] = mutate_enum(self._padding(), PaddingArgs)

    def _mutate_dilation_rate(self, operator=MutationOperators.SYNC_STEP):
        self.args[Conv2DArgs.DILATION_RATE.name] = mutate_tuple(
            self._dilation_rate(), 1, Conv2DLayer.MAX_DILATION, operator
        )

    def _mutate_activation(self):
        self.args[Conv2DArgs.ACTIVATION.name] = mutate_enum(
            self._activation(), Activations
        )

    def mutate(self):
        random.choice(
            [
                self._mutate_activation,
                self._mutate_filters,
                self._mutate_kernel_size,
                self._mutate_padding,
                self._mutate_strides,
                self._mutate_dilation_rate,
            ]
        )()

    def _valid_pad_output_shape(self, input, kernel, stride):
        return ((input - kernel) // stride) + 1

    def _same_pad_output_shape(self, input, stride):
        return ((input - 1) // stride) + 1

    def calc_output_shape(self):
        inp = self._input_size()
        stri = self._strides()
        ks = self._kernel_size()
        fcount = self._filters()
        if self._padding() == PaddingArgs.SAME.value:
            X = self._same_pad_output_shape(inp[0], stri[0])
            Y = self._same_pad_output_shape(inp[1], stri[1])
            self.outputshape.set((X, Y, fcount))
            return
        elif self._padding() == PaddingArgs.VALID.value:
            X = self._valid_pad_output_shape(inp[0], ks[0], stri[0])
            Y = self._valid_pad_output_shape(inp[1], ks[1], stri[1])
            self.outputshape.set((X, Y, fcount))
            return
        self.outputshape.set((0, 0, 0))

    def validate(self):
        if not 0 > self.args[Conv2DArgs.FILTERS.name]:
            return False

        if not self._single_stride() and not self._single_dilation_rate():
            return False

        return True

    def getkeraslayer(self):
        return keras.layers.Conv2D(
            self.args.get(Conv2DArgs.FILTERS.name),
            kernel_size=self.args.get(Conv2DArgs.KERNEL_SIZE.name),
            strides=self.args.get(Conv2DArgs.STRIDES.name),
            input_shape=self.args.get(Conv2DArgs.INPUT_SIZE.name),
            activation=self.args.get(Conv2DArgs.ACTIVATION.name),
            padding=self.args.get(Conv2DArgs.PADDING.name),
            dilation_rate=self.args.get(Conv2DArgs.DILATION_RATE.name),
        )


class MaxPool2DLayer(ModelLayer):
    MUTATABLE_PARAMETERS = 3
    MAX_POOL_SIZE = 7
    MAX_STRIDE = 7

    def __init__(self, pool_size, strides=(1, 1), padding=PaddingArgs.SAME.value):
        super().__init__("MaxPool2D")
        self.args[MaxPool2DArgs.POOL_SIZE.name] = pool_size
        self.args[MaxPool2DArgs.STRIDES.name] = strides
        self.args[MaxPool2DArgs.PADDING.name] = padding

    def getkeraslayer(self):
        return keras.layers.MaxPool2D(
            pool_size=self.args.get(MaxPool2DArgs.POOL_SIZE.name),
            strides=self.args.get(MaxPool2DArgs.STRIDES.name),
            padding=self.args.get(MaxPool2DArgs.PADDING.name),
        )

    def _pool_size(self):
        return self.args[MaxPool2DArgs.POOL_SIZE.name]

    def _strides(self):
        return self.args[MaxPool2DArgs.STRIDES.name]

    def _padding(self):
        return self.args[MaxPool2DArgs.PADDING.name]

    def _mutate_pool_size(self, operator=MutationOperators.SYNC_STEP):
        self.args[MaxPool2DArgs.POOL_SIZE.name] = mutate_tuple(
            self._pool_size(), 1, MaxPool2DLayer.MAX_POOL_SIZE, operator=operator
        )

    def _mutate_strides(self, operator=MutationOperators.SYNC_STEP):
        self.args[MaxPool2DArgs.STRIDES.name] = mutate_tuple(
            self._strides(), 1, MaxPool2DLayer.MAX_STRIDE, operator=operator
        )

    def mutate(self):
        random.choice([self._mutate_pool_size, self._mutate_strides])()


class MaxPool3DLayer(MaxPool2DLayer):
    def __init__(self, pool_size, strides, padding):
        super(MaxPool2DLayer, self).__init__("MaxPool3D")
        super().__init__(pool_size=pool_size, strides=strides, padding=padding)

    def getkeraslayer(self):
        return keras.layers.MaxPool3D(
            pool_size=self.args.get(MaxPool2DArgs.POOL_SIZE.name),
            strides=self.args.get(MaxPool2DArgs.STRIDES.name),
            padding=self.args.get(MaxPool2DArgs.PADDING.name),
        )


class ReshapeLayer(ModelLayer):
    MUTATABLE_PARAMETERS = 0

    def __init__(self, target_shape):
        super().__init__("Reshape")
        self.args[ReshapeArgs.TARGET_SHAPE.name] = target_shape

    def getkeraslayer(self):
        target_shape = self.args.get(ReshapeArgs.TARGET_SHAPE.name)
        return keras.layers.Reshape(target_shape)


class DenseLayer(ModelLayer):
    MUTATABLE_PARAMETERS = 1

    def __init__(self, units, activation):
        super().__init__("Dense")
        self.args[DenseArgs.UNITS.name] = units
        self.args[DenseArgs.ACTIVATION.name] = activation

    def _activation(self):
        return self.args[DenseArgs.ACTIVATION.name]

    def getkeraslayer(self):
        return keras.layers.Dense(
            self.args.get(DenseArgs.UNITS.name),
            activation=self.args.get(DenseArgs.ACTIVATION.name),
        )

    def _mutate_activation(self):
        self.args[DenseArgs.ACTIVATION.name] = mutate_enum(
            self._activation(), Activations
        )

    def mutate(self):
        random.choice([self._mutate_activation])()


class FlattenLayer(ModelLayer):
    MUTATABLE_PARAMETERS = 0

    def __init__(self):
        super().__init__("Flatten")

    def getkeraslayer(self):
        return keras.layers.Flatten()


class Dropout(ModelLayer):
    MUTATABLE_PARAMETERS = 1

    def __init__(self, rate):
        super().__init__("Dropout")
        self.args[DropoutArgs.RATE.name] = rate

    def _rate(self):
        return self.args[DropoutArgs.RATE.name]

    def getkeraslayer(self):
        rate = self.args.get(DropoutArgs.RATE.name)
        return keras.layers.Dropout(rate)

    def _mutate_rate(self):
        self.args[DropoutArgs.RATE.name] = mutate_unit_interval(self._rate(), 0, 1)

    def mutate(self):
        random.choice([self._mutate_rate])()
