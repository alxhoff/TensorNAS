from tensornas.blocktemplates.blockarchitectures import ClassificationBlockArchitecture
import tensorflow as tf

### ENABLE GPU ###
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)
##################

print("##########################################")
print("Testing classification block architecture")
print("##########################################")

model1 = ClassificationBlockArchitecture.ClassificationBlockArchitecture(
    input_tensor_shape, mnist_class_count
)

from tensornas.tools.latexwriter import LatexWriter

lw = LatexWriter()

latex_arch = lw.create_arch(model1)

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
    filename="../conv2d.tflite",
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
