import tensorflow as tf
import tensorflow_datasets as tfds


def load_dataset(dataset, batch_size, dataset_size, train_size, test_size,
                 img_shape):
    # The last 80% of train as train
    ds_train = tfds.load(dataset, split=f'train[-{train_size}%:]',
                         as_supervised=True,
                         shuffle_files=True)
    # The first 10% of train as test
    ds_test = tfds.load(dataset, split=f'train[:{test_size}%]',
                        as_supervised=True,
                        shuffle_files=True)
    ds_train_size = int(dataset_size * train_size / 100)
    ds_test_size = int(dataset_size * test_size / 100)

    def parse_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        image = tf.image.resize(image, img_shape)
        return tf.cast(image, tf.float32) / 255., label

    ds_train = ds_train.map(
        parse_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.take(ds_train_size).cache().batch(batch_size).repeat()
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(
        parse_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.take(ds_test_size).cache().batch(batch_size).repeat()
    return ds_test, ds_test_size, ds_train, ds_train_size
