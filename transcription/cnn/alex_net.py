"""
Implementation of AlexNet(2012)
AlexNet has 8 layers — 5 convolutional and 3 fully-connected.
60M parameters
"""
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from keras.utils.vis_utils import plot_model
from utils.app_setup import *


class AlexNet:

    def __init__(self):
        self._model = Sequential()
        self._input_shape = (IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS)
        self._num_classes = TONE_RANGE
        self._dropout = 0.4

    def create(self):
        # Layer 1.
        self._model.add(Conv2D(filters=96, input_shape=self._input_shape,
                               kernel_size=(11, 11), strides=(4, 4), padding='valid'))
        self._model.add(Activation('relu'))
        self._model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

        # Layer 2.
        self._model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same'))
        self._model.add(Activation('relu'))
        self._model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

        # Layer 3. - 5.
        for _ in range(3):
            self._model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
            self._model.add(Activation('relu'))
        self._model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

        # Fully Connected layer
        self._model.add(Flatten())

        # Layer 6.
        self._model.add(Dense(4096, input_shape=self._input_shape))
        self._model.add(Activation('relu'))
        self._model.add(Dropout(self._dropout))

        # Layer 7.
        self._model.add(Dense(4096))
        self._model.add(Activation('relu'))
        self._model.add(Dropout(self._dropout))

        # Layer 8
        self._model.add(Dense(1000))
        self._model.add(Activation('relu'))
        self._model.add(Dropout(self._dropout))

        self._model.add(Dense(self._num_classes))
        self._model.add(Activation('softmax'))

        self._model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

    def summary(self):
        print(self._model.summary())

    def plot(self, path='./alex_net.png'):
        plot_model(self._model, to_file=path, show_shapes=True)

    def train(self):
        pass
