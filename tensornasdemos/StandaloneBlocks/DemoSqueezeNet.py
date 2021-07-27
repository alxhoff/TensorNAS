from tensornas.blocktemplates.blockarchitectures import SqueezeNetBlockArchitecture
from tensornas.core.util import list_available_blocks

from tensornasdemos.Datasets.MNIST import GetData

images_train, images_test, labels_train, labels_test, input_tensor_shape = GetData()
mnist_class_count = 10

from tensornas.tools.tensorflow.GPU import config_GPU

# enable GPU
config_GPU()

import tensorflow as tf

print("##########################################")
print("Testing Squeeze Net block architecture")
print("##########################################")

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

list_available_blocks()

model = SqueezeNetBlockArchitecture.SqueezeNetBlockArchitecture(
    input_tensor_shape, mnist_class_count
)

# model.print()

print(model.get_ascii_tree())

metrics = model.evaluate(
    train_data=images_train,
    train_labels=labels_train,
    test_data=images_test,
    test_labels=labels_test,
    epochs=2,
    batch_size=32,
    steps=5,
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print(metrics)

model.mutate(verbose=True)

# model.print()

print(model.get_ascii_tree())

metrics = model.evaluate(
    train_data=images_train,
    train_labels=labels_train,
    test_data=images_test,
    test_labels=labels_test,
    epochs=2,
    batch_size=32,
    steps=5,
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print(metrics)

print("Done")
