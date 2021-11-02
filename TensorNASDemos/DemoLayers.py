from TensorNASDemos.Datasets.MNIST import GetData

a, b, c, d, input_tensor_shape = GetData()

from TensorNAS.Layers import Conv2D, Reshape, Dropout, Flatten
from TensorNAS.Layers.Dense import HiddenDense, OutputDense
from TensorNAS.Layers.Pool import MaxPool3D, MaxPool2D

print("Unit testing Layers")

test = Conv2D.Layer(input_tensor_shape)
print(test.get_name())
test.print()
test.mutate()

test = Flatten.Layer(input_tensor_shape)
print(test.get_name())
test.print()
test.mutate()

test = Dropout.Layer(input_tensor_shape)
print(test.get_name())
test.print()
test.mutate()

test = HiddenDense.Layer(input_tensor_shape)
print(test.get_name())
test.print()
test.mutate()
test.print()

test = OutputDense.Layer(input_tensor_shape, 10)
print(test.get_name())
test.print()
test.mutate()
test.print()

test = Reshape.Layer(input_tensor_shape, [1, 784])
print(test.get_name())
test.print()
test.mutate()
test.print()

test = MaxPool2D.Layer(input_tensor_shape)
print(test.get_name())
test.print()
test.mutate()
test.print()

test = MaxPool3D.Layer(input_tensor_shape)
print(test.get_name())
test.print()
test.mutate()
test.print()
