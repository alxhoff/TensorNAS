from tensornas.blocktemplates.blockarchitectures import ClassificationBlockArchitecture
from tensornasdemos.Datasets.MNIST import GetData

import tensorflow as tf

print(tf.__version__)
print(tf.executing_eagerly())

images_train, images_test, labels_train, labels_test, input_tensor_shape = GetData()
mnist_class_count = 10
batch_size = 512
epochs = 2
data_size = 60000
optimizer = "adam"
loss = "sparse_categorical_crossentropy"
metrics = ["accuracy"]


def get_train_data(len, data, labels):
    assert data.shape[0] == labels.shape[0]
    assert len <= data.shape[0]
    import random

    random.seed()
    len = int(len)
    data_len = data.shape[0]
    index = random.randrange(0, data_len + 1 - len)
    return data[index : index + len], labels[index : index + len]


from tensornas.tools.tensorflow.GPU import config_GPU

# enable GPU
config_GPU()

print("##########################################")
print("Testing classification block architecture")
print("##########################################")

model1 = ClassificationBlockArchitecture.ClassificationBlockArchitecture(
    input_tensor_shape, mnist_class_count
)

model1.print()

tr_data, tr_labels = get_train_data(data_size, images_train, labels_train)
te_data, te_labels = get_train_data(data_size / 6, images_test, labels_test)

out_metrics = model1.evaluate(
    train_data=tr_data,
    train_labels=tr_labels,
    test_data=te_data,
    test_labels=te_labels,
    epochs=epochs,
    batch_size=batch_size,
    optimizer=optimizer,
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
    optimizer=optimizer,
    loss=loss,
    metrics=metrics,
)

print(out_metrics)

model2 = ClassificationBlockArchitecture.ClassificationBlockArchitecture(
    input_tensor_shape, mnist_class_count
)

from tensornas.core.crossover import crossover_single_point

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
        optimizer=optimizer,
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
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )
except Exception as e:
    print("Training model 2 failed: {}".format(e))

print(out_metrics)

print("Done")
