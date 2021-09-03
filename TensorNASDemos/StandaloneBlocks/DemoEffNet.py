from TensorNAS.BlockTemplates.BlockArchitectures import EffNetBlockArchitecture

from TensorNASDemos.Datasets.MNIST import GetData

images_train, images_test, labels_train, labels_test, input_tensor_shape = GetData()
mnist_class_count = 10

from TensorNAS.Tools.TensorFlow.GPU import config_GPU

# enable GPU
config_GPU()

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

from TensorNAS.Core.Crossover import crossover_single_point

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
