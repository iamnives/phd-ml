from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow_core.python.keras.losses import sparse_categorical_crossentropy, categorical_crossentropy


def get_model(input_shape=(256, 256, 3), logits=5):
    model = Sequential()

    # 1st Convolutional Layer
    model.add(
        Conv2D(filters=96, input_shape=input_shape, kernel_size=(11, 11),
               strides=(4, 4), padding="valid", activation="relu"))

    # Max Pooling
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))

    # 2nd Convolutional Layer
    model.add(
        Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding="same",
               activation="relu"))

    # Max Pooling
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))

    # 3rd Convolutional Layer
    model.add(
        Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same",
               activation="relu"))

    # 4th Convolutional Layer
    model.add(
        Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same",
               activation="relu"))

    # 5th Convolutional Layer
    model.add(
        Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same",
               activation="relu"))

    # Max Pooling
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))

    # Passing it to a Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(units=9216, activation="relu"))

    # 2nd Fully Connected Layer
    model.add(Dense(units=4096, activation="relu"))

    # 3rd Fully Connected Layer
    model.add(Dense(4096, activation="relu"))

    # Output Layer
    model.add(Dense(logits, activation="softmax"))

    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss=sparse_categorical_crossentropy,
                  metrics=['sparse_categorical_accuracy'])
    return model
