from TensorNAS.BlockTemplates.BlockArchitectures.DSCNNBlockArchitecture import (
    Block as DSCNNBlockArchitecture,
)
from TensorNASDemos.Datasets.SpeechCommands import GetData

import tensorflow as tf

print(tf.__version__)
print(tf.executing_eagerly())

from TensorNAS.Tools.TensorFlow.GPU import config_GPU

# enable GPU
config_GPU()

train_generator, val_generator, input_tensor_shape = GetData()
class_count = 10
batch_size = 512
epochs = 2
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
    val_generator=val_generator,
    epochs=epochs,
    batch_size=batch_size,
    loss=loss,
    metrics=metrics,
    test_name="DemoClassification",
    model_name="Model1",
)

print(out_metrics)

model1.mutate(verbose=True)

print(model1.get_ascii_tree())
