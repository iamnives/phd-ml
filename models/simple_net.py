from tensorflow import keras
from tensorflow.keras import layers


def simple_model(input_shape=(256, 256, 3)):
    image_input = keras.Input(shape=input_shape, name='img_input')

    x1 = layers.Conv2D(3, 3)(image_input)
    x1 = layers.GlobalMaxPooling2D()(x1)

    x2 = layers.Conv2D(3, 3)(image_input)
    x2 = layers.GlobalMaxPooling1D()(x2)

    x = layers.concatenate([x1, x2])

    class_output = layers.Dense(5, name='class_output')(x)

    model = keras.Model(inputs=[image_input],
                        outputs=[class_output])

    model.compile(
        optimizer=keras.optimizers.RMSprop(1e-3),
        loss=[keras.losses.MeanSquaredError(),
              keras.losses.CategoricalCrossentropy(from_logits=True)],
        metrics=[[keras.metrics.MeanAbsolutePercentageError(),
                  keras.metrics.MeanAbsoluteError()],
                 [keras.metrics.CategoricalAccuracy()]])
    return model
