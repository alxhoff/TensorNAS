from TensorNAS.Blocks.BlockArchitectures.DeepAutoEncoderBlockArchitecture import (
    Block as DeepAutoEncoderBlockArchitecture,
)
from TensorNAS.Demos.Datasets.ToyADMOS import GetData, GetTestData

import tensorflow as tf

print(tf.__version__)
print(tf.executing_eagerly())

from TensorNAS.Tools.TensorFlow.GPU import config_GPU

# enable GPU
config_GPU()

data_vector, input_tensor_shape = GetData()
test_data, test_labels = GetTestData()
batch_size = 100
epochs = 1
optimizer = "adam"
loss = "tf.keras.losses.MeanSquaredError()"
metrics = ["accuracy"]
validation_split = 0.1

print("##########################################")
print("Testing Deep Auto Encoder block architecture")
print("##########################################")

model1 = DeepAutoEncoderBlockArchitecture(
    input_shape=input_tensor_shape,
    batch_size=batch_size,
    optimizer=optimizer,
)

model1.print()

out_metrics = model1.evaluate(
    train_data=data_vector,
    test_data=test_data,
    test_labels=test_labels,
    validation_split=validation_split,
    epochs=epochs,
    batch_size=batch_size,
    loss=loss,
    metrics=metrics,
    test_name="DemoClassification",
    model_name="Model1",
)

print("end first eval")
print(out_metrics)
model1.mutate(verbose=True)
print("mutated")
model1.print()


out_metrics = model1.evaluate(
    train_data=data_vector,
    test_data=test_data,
    test_labels=test_labels,
    validation_split=validation_split,
    epochs=epochs,
    batch_size=batch_size,
    loss=loss,
    metrics=metrics,
    test_name="DemoClassification",
    model_name="Model1",
)

print("end second eval")
print(out_metrics)
model1.print()
print(model1.get_ascii_tree())
