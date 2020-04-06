import keras
import os
from transcription.cnn.base import NeuralNet
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from keras.utils.vis_utils import plot_model
from utils.app_setup import *


class DNN(NeuralNet):

    def __init__(self, number_of_units, number_of_layers):
        super().__init__('dnn_{0}'.format(number_of_layers))
        self.model = Sequential()
        self.number_of_units = number_of_units
        self.number_of_layers = number_of_layers
        self.dropout = 0.2
        self.save = None

    def create(self, create_new=False):
        super().create()
        self.model.add(Dense(self.number_of_units, input_shape=(N_BINS, ), kernel_initializer='normal', activation='relu'))
        self.model.add(Dropout(self.dropout))
        for _ in range(self.number_of_layers - 1):
            self.model.add(Dense(self.number_of_units, kernel_initializer='normal', activation='relu'))
            self.model.add(Dropout(self.dropout))
        self.model.add(Dense(NOTE_RANGE, kernel_initializer='normal', activation='sigmoid'))

        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

        checkpoint = ModelCheckpoint(filepath='dnn_{0}.hdf5'.format(self.number_of_layers), verbose=1, save_best_only=False)
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')

        self.callbacks = [checkpoint, earlystop]

    def summary(self):
        super().summary()

    def test(self):
        if os.path.exists(self.model_name):
            pass


