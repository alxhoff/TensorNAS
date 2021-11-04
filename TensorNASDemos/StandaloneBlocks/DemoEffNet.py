from TensorNAS.BlockTemplates.BlockArchitectures.EffNetBlockArchitecture import (
    Block as EffNetBlockArchitecture,
)

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

print("##########################################")
print("Testing classification block architecture")
print("##########################################")

model1 = EffNetBlockArchitecture(
    input_shape=input_tensor_shape,
    class_count=class_count,
    batch_size=batch_size,
    optimizer=optimizer,
)

model1.print()

out_metrics = model1.evaluate(
    train_data=images_train,
    train_labels=labels_train,
    test_data=images_test,
    test_labels=labels_test,
    epochs=epochs,
    batch_size=batch_size,
    steps_per_epoch=steps_per_epoch,
    loss=loss,
    metrics=metrics,
    test_name="DemoEffNet",
    model_name="Model",
)

print(out_metrics)
model1.mutate(verbose=True)
print(model1.get_ascii_tree())

res = model1.evaluate(
    train_data=images_train,
    train_labels=labels_train,
    test_data=images_test,
    test_labels=labels_test,
    epochs=epochs,
    batch_size=batch_size,
    steps_per_epoch=steps_per_epoch,
    loss=loss,
    metrics=metrics,
    test_name="DemoEffNet",
    model_name="Model",
)

print(out_metrics)

model2 = EffNetBlockArchitecture(
    input_shape=input_tensor_shape,
    class_count=class_count,
    batch_size=batch_size,
    optimizer=optimizer,
)

from TensorNAS.Core.Crossover import crossover_single_point

model1, model2 = crossover_single_point(model1, model2)

print(model1.get_ascii_tree())

out_metrics = model1.evaluate(
    train_data=images_train,
    train_labels=labels_train,
    test_data=images_test,
    test_labels=labels_test,
    epochs=epochs,
    batch_size=batch_size,
    steps_per_epoch=steps_per_epoch,
    loss=loss,
    metrics=metrics,
    test_name="DemoEffNet",
    model_name="Model",
)

print(out_metrics)
print("Done")
