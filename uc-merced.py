import tensorflow_datasets as tfds
from tensorflow.keras.utils import plot_model

# Construct a tf.data.Dataset
from models.u_net import get_model, get_siamese_model, get_dual_input_siamese_model

uc_merced, ds_info = tfds.load('uc_merced', split='train', shuffle_files=True, with_info=True)

print(ds_info)

tfds.show_examples(ds_info, uc_merced)

u_net = get_model((256, 256, 3))
u_net_siamese = get_siamese_model((256, 256, 3))
u_net_siamese_dual = get_dual_input_siamese_model(input_shape1=(256, 256, 10), input_shape2=(256, 256, 3))

