from tensorflow.keras import layers, losses, metrics, models, optimizers

OPTIMIZER = 'SGD'
LOSS = 'MeanSquaredError'
METRICS = ['RootMeanSquaredError']


def conv_block(input_tensor, num_filters):
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    return encoder


def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
    return encoder_pool, encoder


def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2),
                                     padding='same')(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    return decoder


def u_net_block(inputs):
    encoder0_pool, encoder0 = encoder_block(inputs, 32)  # 128
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)  # 64
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)  # 32
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)  # 16
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512)  # 8
    center = conv_block(encoder4_pool, 1024)  # center
    decoder4 = decoder_block(center, encoder4, 512)  # 16
    decoder3 = decoder_block(decoder4, encoder3, 256)  # 32
    decoder2 = decoder_block(decoder3, encoder2, 128)  # 64
    decoder1 = decoder_block(decoder2, encoder1, 64)  # 128
    decoder0 = decoder_block(decoder1, encoder0, 32)  # 256
    return decoder0


def get_model(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)  # 256
    decoder0 = u_net_block(inputs)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)

    model = models.Model(inputs=[inputs], outputs=[outputs])

    model.compile(
        optimizer=optimizers.get(OPTIMIZER),
        loss=losses.get(LOSS),
        metrics=[metrics.get(metric) for metric in METRICS])

    return model


def get_siamese_model(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)  # 256
    block0 = u_net_block(inputs)
    block1 = u_net_block(inputs)
    decoder_siamese = layers.concatenate([block0, block1], axis=-1)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder_siamese)

    model = models.Model(inputs=[inputs], outputs=[outputs])

    model.compile(
        optimizer=optimizers.get(OPTIMIZER),
        loss=losses.get(LOSS),
        metrics=[metrics.get(metric) for metric in METRICS])

    return model


def get_dual_input_siamese_model(input_shape1=(256, 256, 3),
                                 input_shape2=(256, 256, 3)):
    inputs0 = layers.Input(shape=input_shape1)  # 256
    inputs1 = layers.Input(shape=input_shape2)  # 256

    block0 = u_net_block(inputs0)
    block1 = u_net_block(inputs1)
    decoder_siamese = layers.concatenate([block0, block1], axis=-1)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder_siamese)

    model = models.Model(inputs=[inputs0, inputs1], outputs=[outputs])

    model.compile(
        optimizer=optimizers.get(OPTIMIZER),
        loss=losses.get(LOSS),
        metrics=[metrics.get(metric) for metric in METRICS])

    return model
