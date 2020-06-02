import keras
import os
from transcription.cnn.base import NeuralNet
from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation, Masking, LSTM as LSTMM
from keras.layers import Dense, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from keras.utils.vis_utils import plot_model
from utils.app_setup import *


class LSTM(NeuralNet):
    def __init__(self, number_of_units, number_of_layers):
        super().__init__('lstm_{0}'.format(number_of_layers))
        self.model = Sequential()
        self.number_of_units = number_of_units
        self.number_of_layers = number_of_layers
        self.dropout = 0.2
        self.save = None

    def create(self, create_new=False):
        super().create(create_new)
        self.model.add(LSTMM(self.number_of_units, input_shape=(LSTM_SAMPLE_SIZE, N_BINS),
                             return_sequences=True, kernel_initializer='normal', activation='tanh'))
        self.model.add(Dropout(self.dropout))
        for _ in range(self.number_of_layers - 1):
            self.model.add(LSTMM(self.number_of_units, return_sequences=True, kernel_initializer='normal', activation='tanh'))
            self.model.add(Dropout(self.dropout))
        self.model.add(Dense(NOTE_RANGE, kernel_initializer='normal', activation='relu'))

        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error', 'accuracy'])

        checkpoint = ModelCheckpoint(filepath='lstm_{0}.hdf5'.format(self.number_of_layers), verbose=1,
                                     save_best_only=False)
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')

        self.callbacks = [checkpoint, earlystop]

    def set_model(self, model):
        self.model = model