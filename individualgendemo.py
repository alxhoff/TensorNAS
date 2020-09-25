import implementedblocks
import demomodels
from tensorflow import keras

##### Training MNIST data
(
    (images_train, labels_train),
    (images_test, labels_test),
) = keras.datasets.mnist.load_data()
input_shape = images_train.shape
images_train = images_train.reshape(
    images_train.shape[0], images_train.shape[1], images_train.shape[2], 1
)
images_test = images_test.reshape(
    images_test.shape[0], images_test.shape[1], images_test.shape[2], 1
)
input_tensor_shape = (images_train.shape[1], images_train.shape[2], 1)
print(input_tensor_shape)
print(labels_train.shape)
images_train = images_train.astype("float32")
images_test = images_test.astype("float32")
images_train /= 255
images_test /= 255
mnist_class_count = 10

# Demo hard-coded models
demomodels.generate_demo_model_jsons()
demo_models = demomodels.generate_demo_model_array()
demo_model_count = 1
#######

model = implementedblocks.BlockArchitecture(input_tensor_shape, mnist_class_count)

print("Done")
