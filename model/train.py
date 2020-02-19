from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import pickle
from tensorflow.keras.utils import Sequence

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
#from vis.utils import utils
import numpy as np
# from scipy.misc import imsave
import imageio
# imageio.imwrite('filename.jpg', array)
import numpy as np

from my_metrics import *
from model import *

train_dir = r'/p/projects/training2006/dataset/challenge5/dataset_rem_lr/train/'
test_dir = r'/p/projects/training2006/dataset/challenge5/dataset_rem_lr/test/'

total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_test = sum([len(files) for r, d, files in os.walk(test_dir)])

image_tensor = layers.Input(shape=(img_height, img_width, img_channels))

network_output = residual_network(image_tensor, 256)

model = Model(inputs=[image_tensor], outputs=[network_output])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1_m, precision_m, recall_m])
print(model.summary())
