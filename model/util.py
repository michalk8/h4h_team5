from keras_augm_layer import *
from resnet import residual_network
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model

IMG_W, IMG_H, IMG_C = 400, 400, 3


# TODO: albumentations
def get_transforms():
    return [
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate(angle=45, p=0.5),
        RandomRotate90(p=0.5),

        RGBShift(p=0.1, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
        RandomBrightness(p=0.999, max_delta=0.1),
        RandomContrast(0.5, 1.5, p=0.9),
        RandomHue(0.5, p=0.9),
        RandomSaturation(0.5, 1.5, p=0.9),

        RandomGaussNoise((10, 50), p=0.99),
        # ToGray(p=0.5),
        JpegCompression(p=0.9, quality_lower=5, quality_upper=99),
    ]


def get_model(img_size, downsampling, apply_transforms=False):
    inp = layers.Input(shape=(IMG_H, IMG_W, IMG_C))

    if apply_transforms:
        transforms = get_transforms()
        x = AugmLayer(transforms, output_dim=(IMG_H, IMG_W, IMG_C))(inp)
    else:
        x = inp

    network_output = residual_network(x, img_size, downsampling)

    model = Model(inputs=[inp], outputs=[network_output])

    return model
