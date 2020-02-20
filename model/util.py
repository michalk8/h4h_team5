from resnet import residual_network
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
import tensorflow as tf

IMG_W, IMG_H, IMG_C = 400, 400, 3


def encode_jpeg(x, quality=None):
    if quality is None:
        return x

    raise NotImplemented('This currently does not work')
    print('Encoding as a JPEG with quality:', quality)
    x = layers.Lambda(lambda image: tf.image.encode_jpeg(image))(x)

    return layers.Lambda(lambda image: tf.image.decode_jpeg(image))(x)


def add_noise(x, sigma=None):
    if sigma is None:
        return x
    print('Adding random noise with stddev:', sigma)
    return layers.Lambda(lambda x: x + tf.random.normal(tf.shape(x), mean=0, stddev=sigma))(x)


def get_model(img_size, downsampling, quality=None, sigma=None):
    inp = layers.Input(shape=(IMG_H, IMG_W, IMG_C))
    x = add_noise(inp, sigma)
    _ = encode_jpeg(x, quality)

    network_output = residual_network(inp, img_size, downsampling)

    model = Model(inputs=[inp], outputs=[network_output])

    return model
