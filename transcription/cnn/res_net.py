"""
Implementation of ResNet (2015)
"""
import numpy as np
import matplotlib.pyplot as plt
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
        self._input_shape = INPUT_SHAPE
        self._input_shape_channels = INPUT_SHAPE_CHANNEL
        self._num_classes = NOTE_RANGE
        self._dropout = 0.5

    def create(self):
        inputs = Input(shape=self._input_shape)
        reshape = Reshape(self._input_shape_channels)(inputs)

        conv = Conv2D(64, (1, BIN_MULTIPLE * NOTE_RANGE), padding='same', activation='relu')(reshape)
        pool = MaxPooling2D(pool_size=(1, 2))(conv)

        for i in range(int(np.log2(BIN_MULTIPLE))):
            bn = BatchNormalization()(pool)
            re = Activation('relu')(bn)
            freq_range = (BIN_MULTIPLE/(2**(i+1))) * NOTE_RANGE
            print(freq_range)
            conv = Conv2D(64, (1, freq_range), padding='same', activation='relu')(re)
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

    def plot_history_accuracy(self, result):
        #model accuracy
        plt.plot(result.history['acc'])
        plt.plot(result.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Valid'], loc='upper left')
        plt.show()

    def plot_model_loss(self, result):
        # model loss
        plt.plot(result.history['loss'])
        plt.plot(result.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Valid'], loc='upper left')
        plt.show()

    def train(self, X_train, y_train, X_validation, y_validation, epochs=1000):
        self.create()
        self._model.compile(loss='binary_crossentropy', optimizer=SGD(momentum=0.9))
        self.summary()

        result = self._model.fit(X_train, y_train, epochs=epochs, validation_data=(X_validation, y_validation))
        print(result.history)

        self.plot_history_accuracy(result)
        self.plot_model_loss(result)