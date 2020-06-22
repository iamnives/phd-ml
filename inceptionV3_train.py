import matplotlib.pyplot as plt

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from utils.data import load_dataset

UC_MERCED_SIZE = 2100
TEST_SIZE = 10
TRAIN_SIZE = 100 - TEST_SIZE
batch_size = 128
epochs = 10
IMG_HEIGHT = 256
IMG_WIDTH = 256
N_CHANNELS = 3
NUM_CLASSES = 21


def main():
    ds_test, ds_test_size, ds_train, ds_train_size = load_dataset(
        dataset='uc_merced', batch_size=batch_size,
        dataset_size=UC_MERCED_SIZE, train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        img_shape=(IMG_WIDTH, IMG_HEIGHT))

    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False,
                             input_shape=(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS))

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 21 classes
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model
    # (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    # train the model on the new data for a few epochs
    hist = model.fit(ds_train,
                     validation_data=ds_test,
                     steps_per_epoch=ds_train_size // batch_size,
                     validation_steps=ds_test_size // batch_size,
                     epochs=epochs)
    # Plot training & validation accuracy values
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # he top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3.
    # We will freeze the bottom N layers
    # and train the remaining top layers.

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    # we train our model again (
    # this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    hist = model.fit(ds_train,
                     validation_data=ds_test,
                     steps_per_epoch=ds_train_size // batch_size,
                     validation_steps=ds_test_size // batch_size,
                     epochs=epochs)
    # Plot training & validation accuracy values
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    main()
