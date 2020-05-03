from models.classification.simple_net import simple_model
from utils.data import load_dataset

UC_MERCED_SIZE = 2100
TEST_SIZE = 10
TRAIN_SIZE = 100 - TEST_SIZE
batch_size = 128
epochs = 100
IMG_HEIGHT = 256
IMG_WIDTH = 256
N_CHANNELS = 3
NUM_CLASSES = 21

ds_test, ds_test_size, ds_train, ds_train_size = load_dataset(
    dataset='uc_merced', batch_size=batch_size,
    dataset_size=UC_MERCED_SIZE, train_size=TRAIN_SIZE,
    test_size=TEST_SIZE,
    img_shape=(IMG_WIDTH, IMG_HEIGHT))

model = simple_model((IMG_HEIGHT, IMG_WIDTH, N_CHANNELS), logits=NUM_CLASSES)

hist = model.fit(ds_train,
                 validation_data=ds_test,
                 steps_per_epoch=ds_train_size // batch_size,
                 validation_steps=ds_test_size // batch_size,
                 epochs=epochs)
