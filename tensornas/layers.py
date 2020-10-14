from tensornas.layerargs import *
from tensornas.mutator import *
from tensorflow import keras
from tensornas.mutator import dimension_mag
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
        return self.dimensions


class ModelLayer(ABC):
    def __init__(self, layer_type, input_shape, args):
        self.layer_type = layer_type
        if args:
            self.args = args
        else:
            self.args = {}

        self.inputshape = LayerShape()
        self.outputshape = LayerShape()

        self.inputshape.set(input_shape)

        self.outputshape.set(self.output_shape())
        self.validate(repair=True)

    def getname(self):
        return self.layer_type.value

    def getargs(self):
        return self.args

    def print(self):
        print(
            "Layer:{} {}-> {}".format(self.getname(), self.inputshape, self.outputshape)
        )
        try:
            arg_list = list(eval(self.getname() + "Args"))
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

    @abstractmethod
    def mutate(self):
        return NotImplementedError

    @abstractmethod
    def output_shape(self):
        pass

    @abstractmethod
    def get_keras_layer(self):
        pass


class Conv2DLayer(ModelLayer):
    MUTATABLE_PARAMETERS = 6
    MAX_FILTER_COUNT = 128
    MAX_KERNEL_DIMENSION = 7
    MAX_STRIDE = 7
    MAX_DILATION = 5

    def _filters(self):
        return self.args[Conv2DArgs.FILTERS.value]

    def _kernel_size(self):
        return self.args[Conv2DArgs.KERNEL_SIZE.value]

    def _strides(self):
        return self.args[Conv2DArgs.STRIDES.value]

    def _input_size(self):
        return self.args[Conv2DArgs.INPUT_SHAPE.value]

    def _padding(self):
        return self.args[Conv2DArgs.PADDING.value]

    def _dilation_rate(self):
        return self.args[Conv2DArgs.DILATION_RATE.value]

    def _activation(self):
        return self.args[Conv2DArgs.ACTIVATION.value]

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
        self.args[Conv2DArgs.FILTERS.value] = mutate_int(
            self._filters(), 1, Conv2DLayer.MAX_FILTER_COUNT, operator
        )

    def _mutate_kernel_size(self, operator=MutationOperators.SYNC_STEP):
        self.args[Conv2DArgs.KERNEL_SIZE.value] = mutate_tuple(
            self._kernel_size(), 1, Conv2DLayer.MAX_KERNEL_DIMENSION, operator
        )

    def _mutate_strides(self, operator=MutationOperators.SYNC_STEP):
        self.args[Conv2DArgs.STRIDES.value] = mutate_tuple(
            self._strides(), 1, Conv2DLayer.MAX_STRIDE, operator
        )

    def _mutate_padding(self):
        self.args[Conv2DArgs.PADDING.value] = mutate_enum(self._padding(), PaddingArgs)

    def _mutate_dilation_rate(self, operator=MutationOperators.SYNC_STEP):
        self.args[Conv2DArgs.DILATION_RATE.value] = mutate_tuple(
            self._dilation_rate(), 1, Conv2DLayer.MAX_DILATION, operator
        )

    def _mutate_activation(self):
        self.args[Conv2DArgs.ACTIVATION.value] = mutate_enum(
            self._activation(), Activations
        )

    def repair(self):
        # TODO
        pass

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

    def validate(self, repair=True):
        if not self._filters() > 0:
            return False

        if not self._single_stride() and not self._single_dilation_rate():
            return False

        if not self._strides()[0] > 0 or not self._strides()[1] > 0:
            return False

        return True

    def output_shape(self):
        return Conv2DLayer.conv2Doutputshape(
            input_size=self._input_size(),
            stride=self._strides(),
            kernel_size=self._kernel_size(),
            filter_count=self._filters(),
            padding=self._padding(),
        )

    @staticmethod
    def _valid_pad_output_shape(input, kernel, stride):
        return ((input - kernel) // stride) + 1

    @staticmethod
    def _same_pad_output_shape(input, stride):
        return ((input - 1) // stride) + 1

    @staticmethod
    def conv2Doutputshape(input_size, stride, kernel_size, filter_count, padding):
        if padding == PaddingArgs.SAME.value:
            X = Conv2DLayer._same_pad_output_shape(input_size[0], stride[0])
            Y = Conv2DLayer._same_pad_output_shape(input_size[1], stride[1])
            return X, Y, filter_count
        elif padding == PaddingArgs.VALID.value:
            X = Conv2DLayer._valid_pad_output_shape(
                input_size[0], kernel_size[0], stride[0]
            )
            Y = Conv2DLayer._valid_pad_output_shape(
                input_size[1], kernel_size[1], stride[1]
            )
            return X, Y, filter_count
        return 0, 0, 0

    def get_keras_layer(self):
        return keras.layers.Conv2D(
            self.args.get(Conv2DArgs.FILTERS.value),
            kernel_size=self.args.get(Conv2DArgs.KERNEL_SIZE.value),
            strides=self.args.get(Conv2DArgs.STRIDES.value),
            input_shape=self.args.get(Conv2DArgs.INPUT_SHAPE.value),
            activation=self.args.get(Conv2DArgs.ACTIVATION.value),
            padding=self.args.get(Conv2DArgs.PADDING.value),
            dilation_rate=self.args.get(Conv2DArgs.DILATION_RATE.value),
        )


class MaxPool2DLayer(ModelLayer):
    MUTATABLE_PARAMETERS = 3
    MAX_POOL_SIZE = 7
    MAX_STRIDE = 7

    def _pool_size(self):
        return self.args[MaxPool2DArgs.POOL_SIZE.value]

    def _strides(self):
        return self.args[MaxPool2DArgs.STRIDES.value]

    def _padding(self):
        return self.args[MaxPool2DArgs.PADDING.value]

    def _mutate_pool_size(self, operator=MutationOperators.SYNC_STEP):
        self.args[MaxPool2DArgs.POOL_SIZE.value] = mutate_tuple(
            self._pool_size(), 1, MaxPool2DLayer.MAX_POOL_SIZE, operator=operator
        )

    def _mutate_strides(self, operator=MutationOperators.SYNC_STEP):
        self.args[MaxPool2DArgs.STRIDES.value] = mutate_tuple(
            self._strides(), 1, MaxPool2DLayer.MAX_STRIDE, operator=operator
        )

    @staticmethod
    def _valid_pad_output_shape(input, pool, stride):
        return ((input - pool) // stride) + 1

    @staticmethod
    def _same_pad_output_shape(input, pool, stride):
        return MaxPool2DLayer._valid_pad_output_shape(input, pool, stride) + (
            1 if ((input - pool) % stride) else 0
        )

    def repair(self):
        for x, val in enumerate(self._strides()):
            if not val > 0:
                self.args[MaxPool2DArgs.STRIDES.value][x] = 1

        for x, val in enumerate(self._pool_size()):
            if not val > 0:
                self.args[MaxPool2DArgs.POOL_SIZE.value][x] = 1

    def mutate(self):
        random.choice([self._mutate_pool_size, self._mutate_strides])()

    def validate(self, repair=True):
        if not self._strides()[0] > 0 or not self._strides()[1] > 0:
            if repair:
                while not self.validate(repair):
                    self.repair()
            else:
                return False
        if not self._pool_size()[0] > 0 or not self._pool_size()[1] > 0:
            if repair:
                while not self.validate(repair):
                    self.repair()
            else:
                return False
        return True

    def output_shape(self):
        inp = self.inputshape.get()
        pool = self._pool_size()
        stri = self._strides()
        if self._padding() == PaddingArgs.SAME.value:
            x = self._same_pad_output_shape(inp[0], pool[0], stri[0])
            y = self._same_pad_output_shape(inp[1], pool[1], stri[1])
            return x, y, inp[2]
        elif self._padding() == PaddingArgs.VALID.value:
            x = self._valid_pad_output_shape(inp[0], pool[0], stri[0])
            y = self._valid_pad_output_shape(inp[1], pool[1], stri[1])
            return x, y, inp[2]
        return 0, 0, 0

    def get_keras_layer(self):
        return keras.layers.MaxPool2D(
            pool_size=self.args.get(MaxPool2DArgs.POOL_SIZE.value),
            strides=self.args.get(MaxPool2DArgs.STRIDES.value),
            padding=self.args.get(MaxPool2DArgs.PADDING.value),
        )


class MaxPool3DLayer(MaxPool2DLayer):
    def repair(self):
        # TODO
        pass

    def mutate(self):
        # TODO
        pass

    def validate(self, repair=True):
        if (
            not self._strides()[0] > 0
            or not self._strides()[1] > 0
            or not self._strides()[2] > 0
        ):
            return False
        if (
            not self._pool_size()[0] > 0
            or not self._pool_size()[1] > 0
            or not self._pool_size()[2] > 0
        ):
            return False
        return True

    def output_shape(self):
        # TODO
        return self.inputshape

    def get_keras_layer(self):
        return keras.layers.MaxPool3D(
            pool_size=self.args.get(MaxPool2DArgs.POOL_SIZE.value),
            strides=self.args.get(MaxPool2DArgs.STRIDES.value),
            padding=self.args.get(MaxPool2DArgs.PADDING.value),
        )


class ReshapeLayer(ModelLayer):
    MUTATABLE_PARAMETERS = 0

    def _target_shape(self):
        return self.args.get(ReshapeArgs.TARGET_SHAPE.value, self.inputshape.get())

    def _mutate_target_shape(self):
        self.args[ReshapeArgs.TARGET_SHAPE.value] = mutate_dimension(
            self._target_shape()
        )

    def repair(self):
        self.inputshape.set(self.outputshape.get())
        self._mutate_target_shape()

    def mutate(self):
        self._mutate_target_shape()

    def validate(self, repair=True):
        input_mag = dimension_mag(list(self.inputshape.get()))
        output_mag = dimension_mag(list(self.output_shape()))

        if not input_mag == output_mag:
            if repair:
                while not input_mag == output_mag:
                    self.repair()
                    input_mag = dimension_mag(list(self.inputshape.get()))
                    output_mag = dimension_mag(list(self.output_shape()))
            else:
                return False
        return True

    def output_shape(self):
        return self._target_shape()

    def get_keras_layer(self):
        target_shape = self.args.get(ReshapeArgs.TARGET_SHAPE.value)
        return keras.layers.Reshape(target_shape)


class DenseLayer(ModelLayer):
    def _activation(self):
        return self.args[DenseArgs.ACTIVATION.value]

    def _units(self):
        return self.args[DenseArgs.UNITS.value]

    def _mutate_activation(self):
        self.args[DenseArgs.ACTIVATION.value] = mutate_enum(
            self._activation(), Activations
        )

    def repair(self):
        # TODO
        pass

    def mutate(self):
        random.choice([self._mutate_activation])()

    def validate(self, repair=True):
        # Dense layers take in a 1D tensor array, ie. previous layer should be a flatten layer
        if not len(self.inputshape.get()) == 1:
            return False

        return True

    def output_shape(self):
        return (self._units(),)

    def get_keras_layer(self):
        return keras.layers.Dense(
            self.args.get(DenseArgs.UNITS.value),
            activation=self.args.get(DenseArgs.ACTIVATION.value),
        )


class HiddenDenseLayer(DenseLayer):
    MAX_UNITS = 256
    MUTATABLE_PARAMETERS = 2

    def _mutate_units(self):
        self.args[DenseArgs.UNITS.value] = mutate_int(
            self._unit(), 1, HiddenDenseLayer.MAX_UNITS
        )

    def repair(self):
        # TODO
        pass

    def mutate(self):
        random.choice([self._mutate_activation, self._mutate_units])()

    def validate(self, repair=True):
        super().validate()

    def output_shape(self):
        super().output_shape()

    def get_keras_layer(self):
        super().get_keras_layer()


class OutputDenseLayer(DenseLayer):
    MUTATABLE_PARAMETERS = 1

    def repair(self):
        # TODO
        pass

    def mutate(self):
        super().mutate()

    def validate(self, repair=True):
        super().validate()

    def output_shape(self):
        super().output_shape()

    def get_keras_layer(self):
        super().get_keras_layer()


class FlattenLayer(ModelLayer):
    MUTATABLE_PARAMETERS = 0

    def repair(self):
        # TODO
        pass

    def mutate(self):
        pass

    def validate(self, repair=True):
        return True

    def output_shape(self):
        return (dimension_mag(self.inputshape.get()),)

    def get_keras_layer(self):
        return keras.layers.Flatten()


class DropoutLayer(ModelLayer):
    MUTATABLE_PARAMETERS = 1
    MAX_RATE = 0.5

    def _rate(self):
        return self.args[DropoutArgs.RATE.value]

    def _mutate_rate(self):
        self.args[DropoutArgs.RATE.value] = mutate_unit_interval(
            self._rate(), 0, DropoutLayer.MAX_RATE
        )

    def repair(self):
        # TODO
        pass

    def mutate(self):
        random.choice([self._mutate_rate])()

    def validate(self, repair=True):
        return True

    def output_shape(self):
        return self.inputshape.get()

    def get_keras_layer(self):
        rate = self.args.get(DropoutArgs.RATE.value)
        return keras.layers.Dropout(rate)
