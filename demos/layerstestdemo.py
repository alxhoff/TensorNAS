from demos.mnistdemoinput import input_tensor_shape

from tensornas.layers import Conv2D, Reshape, Dropout, Flatten
from tensornas.layers.Dense import HiddenDense, OutputDense
from tensornas.layers.MaxPool import MaxPool3D, MaxPool2D

print("Unit testing layers")

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
