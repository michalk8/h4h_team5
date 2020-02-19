from resnet import residual_network
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model

IMG_W, IMG_H, IMG_C = 400, 400, 3


def get_transforms():
    return []


def get_model(img_size, downsampling, apply_transforms=True):
    inp = layers.Input(shape=(IMG_H, IMG_W, IMG_C))

    if apply_transforms:
        transforms = get_transforms()
        raise NotImplemented('TODO: implement')
        x = inp
    else:
        x = inp

    network_output = residual_network(x, img_size, downsampling)

    model = Model(inputs=[inp], outputs=[network_output])

    return model
