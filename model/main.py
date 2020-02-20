#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pickle
import numpy as np
import random
import argparse
import tensorflow.keras.layers as layers
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from resnet import residual_network
from my_metrics import *
from util import *

_resise_methods = dict(
    area          = tf.image.ResizeMethod.AREA,
    bilinear      = tf.image.ResizeMethod.BILINEAR,
    bicubic       = tf.image.ResizeMethod.BICUBIC,
    # gaussian      = tf.image.ResizeMethod.GAUSSIAN,
    # lanczos3      = tf.image.ResizeMethod.LANCZOS3,
    # lanczos5      = tf.image.ResizeMethod.LANCZOS5,
    # mitchellcubic = tf.image.ResizeMethod.MITCHELLCUBIC,
    nearest       = tf.image.ResizeMethod.NEAREST_NEIGHBOR
)


EPOCHS = 5
BATCH_SIZE = 16
IMG_W, IMG_H, IMG_C = 400, 400, 3

seed = 42
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

tf.set_random_seed(seed)
sess = tf.Session(graph=tf.get_default_graph())
K.set_session(sess)


def main(args):
    train_dir = os.path.join(args.dataset_root, 'train')
    test_dir = os.path.join(args.dataset_root, 'test')
    total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
    total_test = sum([len(files) for r, d, files in os.walk(test_dir)])

    model = get_model(args.img_size, _resise_methods[args.downsampling], args.jpeg_quality, args.sigma)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1_m, precision_m, recall_m])
    print(model.summary())

    image_gen_train = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=359,
        horizontal_flip=True,
        vertical_flip=True,
    )
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation

    train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                         directory=train_dir,
                                                         shuffle=True,
                                                         target_size=(IMG_H, IMG_W),
                                                         class_mode='categorical')
    val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                                  directory=test_dir,
                                                                  target_size=(IMG_H, IMG_W),
                                                                  class_mode='categorical')

    files_per_class = []
    input_foldr = train_dir
    for folder in os.listdir(input_foldr):
        if not os.path.isfile(folder):
            files_per_class.append(len(os.listdir(input_foldr + '/' + folder)))

    total_files = sum(files_per_class)
    class_weights = {}
    for i in range(len(files_per_class)):
        class_weights[i] = 1 - (float(files_per_class[i]) / total_files)

    tensorboard_callback = tf.keras.callbacks.TensorBoard('logs', write_images=True)
    csv_callback = tf.keras.callbacks.callbacks.CSVLogger('data_{}_{}_{}_{}.csv'.format(args.img_sze, args.downsampling,
                                                                                        args.sigma, args.jpeg_quality), separator=',', append=False)

    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=total_train // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_data_gen,
        validation_steps=total_test // BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[tensorboard_callback, csv_callback]
    )

    with open('trainHistoryDict_{}_{}_{}_{}.pickle'.format(args.img_size, args.downsampling, args.sigma,
                                                           args.jpeg_quality), 'wb') as fout:
        pickle.dump(history.history, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root', type=str)
    parser.add_argument('img_size', type=int)
    parser.add_argument('downsampling', type=str)
    parser.add_argument('--sigma', type=float, default=None)
    parser.add_argument('--jpeg-quality', type=int, default=None)

    main(parser.parse_args())
