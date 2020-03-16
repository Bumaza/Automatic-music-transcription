"""
Implementation of ResNet (2015)
"""

from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, add
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
from utils.app_setup import *


class ResNet:

    def __init__(self):
        self._model = Model()
        self._input_shape = (IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS)
        self._num_classes = TONE_RANGE
        self._dropout = 0.5

    def create(self):
        inputs = Input(shape=self._input_shape)

        conv = Conv2D(64, (), padding='same', activation='relu')(inputs)
        pool = MaxPooling2D(pool_size=(1, 2))(conv)

        for i in range(2):
            bn = BatchNormalization()(pool)
            re = Activation('relu')(bn)
            conv = Conv2D(64, (), padding='same', activation='relu')(re)
            ad = add([pool, conv])
            pool = MaxPooling2D(pool_size=(1, 2))(ad)

        flattened = Flatten()(pool)
        fc = Dense(1024, activation='relu')(flattened)
        do = Dropout(self._dropout)(fc)
        fc = Dense(512, activation='relu')(do)
        do = Dropout(self._dropout)(fc)
        outputs = Dense(self._num_classes, activation='sigmoid')(do)

        self._model = Model(inputs=inputs, outputs=outputs)

    def summary(self):
        print(self._model.summary())

    def plot(self, path='./res_net.png'):
        plot_model(self._model, to_file=path, show_shapes=True)

    def train(self):
        pass

