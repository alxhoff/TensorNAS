import tensorflow
from tensorflow import keras


directory="/Volumes/DS/Priya_Desktop/VWW_MicroNet/VWW_Dataset"


datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator()
train_it = datagen.flow_from_directory(f'{directory}/train', class_mode='binary',color_mode='grayscale')
test_it = datagen.flow_from_directory(f'{directory}/val', class_mode='binary',color_mode='grayscale')
input_tensor_shape = train_it.image_shape
print(input_tensor_shape)
#batchy i snumber of batches
#batchX, batchy = train_it.next()
train, batch_train = train_it.next()
print(train.shape)
images_train = train.astype("float32")
test, batch_test= test_it.next()
images_test= test.astype("float32")
images_train /= 255
images_test /= 255
mnist_class_count = 2
print('Batch shape=%s, min=%.3f, max=%.3f' % (train.shape, batch_train.min(), batch_train.max()))
'''
def load_images():
    tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    )
Load_Image()
'''