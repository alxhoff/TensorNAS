from TensorNAS.BlockTemplates.BlockArchitectures.SqueezeNetBlockArchitecture import (
    Block as SqueezeNetBlockArchitecture,
)
from TensorNAS.Tools.Util import list_available_blocks

from TensorNASDemos.Datasets.MNIST import GetData

images_train, images_test, labels_train, labels_test, input_tensor_shape = GetData()
class_count = 10
batch_size = 56
steps_per_epoch = len(images_train) // batch_size
validation_steps = len(images_test) // batch_size
epochs = 1
optimizer = "adam"
loss = "tf.keras.metrics.sparse_categorical_crossentropy"
metrics = ["accuracy"]

from TensorNAS.Tools.TensorFlow.GPU import config_GPU

# enable GPU
config_GPU()

import tensorflow as tf

print("##########################################")
print("Testing Squeeze Net block architecture")
print("##########################################")

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

list_available_blocks()

model = SqueezeNetBlockArchitecture(
    input_shape=input_tensor_shape,
    class_count=class_count,
    batch_size=batch_size,
    optimizer=optimizer,
)

model.print()
print(model.get_ascii_tree())

out_metrics = model.evaluate(
    train_data=images_train,
    train_labels=labels_train,
    test_data=images_test,
    test_labels=labels_test,
    epochs=epochs,
    batch_size=batch_size,
    steps_per_epoch=steps_per_epoch,
    loss=loss,
    metrics=metrics,
    test_name="DemoSqueezeNet",
    model_name="Model",
)

print(out_metrics)
model.mutate(verbose=True)
model.print()
print(model.get_ascii_tree())

out_metrics = model.evaluate(
    train_data=images_train,
    train_labels=labels_train,
    test_data=images_test,
    test_labels=labels_test,
    epochs=epochs,
    batch_size=batch_size,
    steps_per_epoch=steps_per_epoch,
    loss=loss,
    metrics=metrics,
    test_name="DemoSqueezeNet",
    model_name="Model",
)

print(out_metrics)
print("Done")
