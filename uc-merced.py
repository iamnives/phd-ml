import tensorflow as tf
import tensorflow_datasets as tfds
from models.classification.simple_net import simple_model

UC_MERCED_SIZE = 2100
TEST_SIZE = 0.1
TRAIN_SIZE = 1 - TEST_SIZE
batch_size = 128
epochs = 100
IMG_HEIGHT = 256
IMG_WIDTH = 256
N_CHANNELS = 3
NUM_CLASSES = 21

# The last 80% of train as train
ds_train = tfds.load('uc_merced', split='train[-90%:]', as_supervised=True,
                     shuffle_files=True)

# The first 10% of train as test
ds_test = tfds.load('uc_merced', split='train[:10%]', as_supervised=True,
                    shuffle_files=True)

ds_train_size = int(UC_MERCED_SIZE * TRAIN_SIZE)
ds_test_size = int(UC_MERCED_SIZE * TEST_SIZE)


def parse_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.expand_dims(image, 0)
    return tf.cast(image, tf.float32) / 255., label


ds_train = ds_train.map(
    parse_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.take(ds_train_size).cache().repeat()
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(
    parse_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.take(ds_test_size).cache().repeat()

model = simple_model((IMG_HEIGHT, IMG_WIDTH, N_CHANNELS), logits=NUM_CLASSES)

hist = model.fit(ds_train,
                 validation_data=ds_test,
                 steps_per_epoch=ds_train_size,
                 validation_steps=ds_test_size,
                 epochs=epochs)
