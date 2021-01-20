from tensornas.blocktemplates.blockarchitectures import EffNetBlockArchitecture
from demos.DemoMNISTInput import *

print("##########################################")
print("Testing classification block architecture")
print("##########################################")

model1 = EffNetBlockArchitecture.EffNetBlockArchitecture(
    input_tensor_shape, mnist_class_count
)

model1.print()

metrics = model1.evaluate(
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
    filename="effnet.tflite",
)

print(metrics)

model1.mutate(verbose=True)

print(model1.get_ascii_tree())

metrics = model1.evaluate(
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

model2 = EffNetBlockArchitecture.EffNetBlockArchitecture(
    input_tensor_shape, mnist_class_count
)

from tensornas.core.crossover import crossover_single_point

model1, model2 = crossover_single_point(model1, model2)

print(model1.get_ascii_tree())

metrics = model1.evaluate(
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
