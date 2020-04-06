import os
import numpy as np
import matplotlib.pyplot as plt
from utils.app_setup import MODELS_DIR
from keras.models import load_model


class NeuralNet(object):
    def __init__(self, name):
        print('Neural Network')
        self.name = name
        self.model_name = '{0}.hdf5'.format(self.name)
        self.model = None
        self.callbacks = []
        self.batch_size = 512
        self.epochs = 1000
        self.save = None

    def create(self, create_new=False):
        if not create_new and os.path.exists(self.model_name):
            return self.load()

    def load(self):
        self.model = load_model(self.model_name)

    def summary(self):
        print(self.model.summary())

    def train(self, X_train, y_train, X_val, y_val):
        if self.model is None:
            return
        self.save = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs,
                                   validation_data=(X_val, y_val), verbose=1, callbacks=self.callbacks)
        self.plot_result()

    def predict(self, X, y=None):
        if self.model is None:
            return
        y_pred = self.model.predict(X, batch_size=self.batch_size, verbose=1)
        self.plot_predict(self.postprocess(y_pred))

        if y is not None:
            self.plot_predict(y, True)

    def postprocess(self, y_pred):
        y_pred = np.array(y_pred).round()
        y_pred[y_pred > 1] = 1

        for note in range(y_pred.shape[1]):
            for frame in range(2, y_pred.shape[0] - 3):

                if y_pred[frame-1:frame+3, note] == [1, 0, 0, 1]:
                    y_pred[frame, note], y_pred[frame + 1, note] = 1, 1

                if y_pred[frame-2:frame+4, note] == [0, 0, 1, 1, 0, 0]:
                    y_pred[frame, note], y_pred[frame + 1, note] = 0, 0

                if y_pred[frame-1:frame+3, note] == [0, 1, 0, 0]:
                    y_pred[frame, note] = 0

                if y_pred[frame-1:frame+3, note] == [1, 0, 1, 1]:
                    y_pred[frame, note] = 1

        return y_pred

    def plot_result(self):
        if self.save is None:
            return

    def plot_predict(self, y, truth=False):
        fig = plt.figure(figsize=(20, 5))
        plt.title('Ground Truth' if truth else 'Predict')
        plt.imshow(y.T, aspect='auto')
        plt.gca().invert_yaxis()
        fig.axes[0].set_xlabel('window')
        fig.axes[0].set_ylabel('note (MIDI code')
        plt.show()
