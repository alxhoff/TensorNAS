from tensornas.blocktemplates.blockarchitectures import ClassificationBlockArchitecture

from tensornasdemos.Datasets.MNIST import GetData

images_train, images_test, labels_train, labels_test, input_tensor_shape = GetData()
mnist_class_count = 10

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

model2 = ClassificationBlockArchitecture.ClassificationBlockArchitecture(
    input_tensor_shape, mnist_class_count
)

from tensornas.core.crossover import crossover_single_point

model1, model2 = crossover_single_point(model1, model2)

print(model1.get_ascii_tree())
print(model2.get_ascii_tree())

try:
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
except Exception as e:
    print("Training model 1 failed: {}".format(e))

try:
    metrics = model2.evaluate(
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
except Exception as e:
    print("Training model 2 failed: {}".format(e))

print(metrics)

print("Done")
