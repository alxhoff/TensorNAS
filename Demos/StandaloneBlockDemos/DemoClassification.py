from TensorNAS.BlockTemplates.BlockArchitectures.ClassificationBlockArchitecture import (
    Block as ClassificationBlockArchitecture,
)
from TensorNASDemos.Datasets.MNIST import GetData

import tensorflow as tf

print(tf.__version__)
print(tf.executing_eagerly())

tr_data, te_data, tr_labels, te_labels, input_tensor_shape = GetData()
mnist_class_count = 10
batch_size = 512
epochs = 2
data_size = 60000
optimizer = "adam"
loss = "tf.keras.metrics.sparse_categorical_crossentropy"
metrics = ["accuracy"]

from TensorNAS.Tools.TensorFlow.GPU import config_GPU

# enable GPU
config_GPU()

print("##########################################")
print("Testing classification block architecture")
print("##########################################")

model1 = ClassificationBlockArchitecture(
    input_shape=input_tensor_shape,
    class_count=mnist_class_count,
    batch_size=batch_size,
    optimizer=optimizer,
)

model1.print()

out_metrics = model1.evaluate(
    train_data=tr_data,
    train_labels=tr_labels,
    test_data=te_data,
    test_labels=te_labels,
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

out_metrics = model1.evaluate(
    train_data=tr_data,
    train_labels=tr_labels,
    test_data=te_data,
    test_labels=te_labels,
    epochs=epochs,
    batch_size=batch_size,
    loss=loss,
    metrics=metrics,
)

print(out_metrics)

model2 = ClassificationBlockArchitecture(
    input_shape=input_tensor_shape,
    class_count=mnist_class_count,
    batch_size=batch_size,
    optimizer=optimizer,
)

from TensorNAS.Core.Crossover import crossover_single_point

model1, model2 = crossover_single_point(model1, model2)
print(model1.get_ascii_tree())
print(model2.get_ascii_tree())

try:
    out_metrics = model1.evaluate(
        train_data=tr_data,
        train_labels=tr_labels,
        test_data=te_data,
        test_labels=te_labels,
        epochs=epochs,
        batch_size=batch_size,
        loss=loss,
        metrics=metrics,
    )
except Exception as e:
    print("Training model 1 failed: {}".format(e))

try:
    out_metrics = model2.evaluate(
        train_data=tr_data,
        train_labels=tr_labels,
        test_data=te_data,
        test_labels=te_labels,
        epochs=epochs,
        batch_size=batch_size,
        loss=loss,
        metrics=metrics,
    )
except Exception as e:
    print("Training model 2 failed: {}".format(e))

print(out_metrics)

print("Done")
