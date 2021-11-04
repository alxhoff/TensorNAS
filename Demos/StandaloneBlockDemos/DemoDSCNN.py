from TensorNAS.BlockTemplates.BlockArchitectures.DSCNNBlockArchitecture import (
    Block as DSCNNBlockArchitecture,
)
from Demos.Datasets.SpeechCommands import GetData

import tensorflow as tf

print(tf.__version__)
print(tf.executing_eagerly())

from TensorNAS.Tools.TensorFlow.GPU import config_GPU

# enable GPU
config_GPU()

train_generator, val_generator, input_tensor_shape, train_len, val_len = GetData()
class_count = 12
batch_size = 1
steps_per_epoch = train_len // batch_size
validation_steps = val_len // batch_size
epochs = 1
optimizer = "adam"
loss = "tf.keras.metrics.sparse_categorical_crossentropy"
metrics = ["accuracy"]

print("##########################################")
print("Testing classification block architecture")
print("##########################################")

model1 = DSCNNBlockArchitecture(
    input_shape=input_tensor_shape,
    class_count=class_count,
    batch_size=batch_size,
    optimizer=optimizer,
)

model1.print()

out_metrics = model1.evaluate(
    train_generator=train_generator,
    validation_generator=val_generator,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    batch_size=batch_size,
    loss=loss,
    metrics=metrics,
    test_name="DemoClassification",
    model_name="Model1",
)

print(out_metrics)
model1.mutate(verbose=True)
model1.print()

out_metrics = model1.evaluate(
    train_generator=train_generator,
    validation_generator=val_generator,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    batch_size=batch_size,
    loss=loss,
    metrics=metrics,
    test_name="DemoClassification",
    model_name="Model1",
)

print(out_metrics)
model1.print()
print(model1.get_ascii_tree())
