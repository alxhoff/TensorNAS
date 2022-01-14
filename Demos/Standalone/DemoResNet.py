from TensorNAS.Blocks.BlockArchitectures.ResNetBlockArchitecture import (
    Block as ResNetBlockArchitecture,
)
from TensorNAS.Tools import list_available_blocks

from TensorNAS.Demos.Datasets.Cifar10 import GetData

images_train, images_test, labels_train, labels_test, input_tensor_shape = GetData()
class_count = 10
batch_size = 100
steps_per_epoch = len(images_train) // batch_size
validation_steps = len(images_test) // batch_size
epochs = 1
optimizer = "adam"
loss = "tf.nn.softmax_cross_entropy_with_logits"
metrics = ["accuracy"]

from TensorNAS.Tools.TensorFlow.GPU import config_GPU

# enable GPU
config_GPU()

print("##########################################")
print("Testing Res Net block architecture")
print("##########################################")

list_available_blocks()

model = ResNetBlockArchitecture(
    input_shape=input_tensor_shape,
    class_count=class_count,
    batch_size=batch_size,
    optimizer=optimizer,
)

model.print()

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
    test_name="DemoResNet",
    model_name="Model",
)

print(out_metrics)
model.mutate(verbose=True)
print(model.get_ascii_tree())
model.print()

out_metrics = model.evaluate(
    train_data=images_train,
    train_labels=images_train,
    test_data=images_test,
    test_labels=labels_test,
    epochs=epochs,
    batch_size=batch_size,
    steps_per_epoch=steps_per_epoch,
    loss=loss,
    metrics=metrics,
    test_name="DemoResNet",
    model_name="Model",
)

print(out_metrics)
print("Done")
