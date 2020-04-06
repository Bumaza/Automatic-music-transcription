from os import listdir, path
from utils.audio_analysis import *
from utils.midi_utils import *
from utils.app_setup import *
from PIL import Image
import librosa
import os
import pretty_midi
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.regularizers import l2
from keras import callbacks
from keras.callbacks import History, ModelCheckpoint, EarlyStopping


def dnn(X, y, X_val, y_val, X_test, y_test):

    number_of_units, number_of_layers = 256, 3

    model = Sequential()
    history = History()

    model.add(Dense(number_of_units, input_shape=(N_BINS, ), kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    for i in range(number_of_layers-1):
        model.add(Dense(number_of_units, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.2))
    model.add(Dense(NOTE_RANGE, kernel_initializer='normal', activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

    checkpointer = ModelCheckpoint(filepath=os.path.join(), verbose=1, save_best_only=False)
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')

    save = model.fit(X, y, batch_size=100, epochs=1000, validation_data=(X_val, y_val), verbose=1,
                     callbacks=[checkpointer, early])
    print(save)

    pred = model.predict(X_test)
    for i in [400, 800, 1400]:
        print(max(pred[i]), sum(pred[i]))
        print(pred[i])
        print(y_test[i])

    fig = plt.figure(figsize=(20, 5))
    plt.imshow(pred.T, aspect='auto')
    plt.gca().invert_yaxis()
    fig.axes[0].set_xlabel('window')
    fig.axes[0].set_ylabel('note (MIDI code')
    plt.show()


def predict_load(X, y):
    model = load_model(os.path.join(MODELS_DIR, 'weights.hdf5'))

    pred = model.predict(X)
    for i in [400, 800, 1400]:
        print(max(pred[i]), sum(pred[i]))
        print(pred[i])
        print(y[i])

    fig = plt.figure(figsize=(20, 5))
    plt.imshow(pred.T, aspect='auto')
    plt.gca().invert_yaxis()
    fig.axes[0].set_xlabel('window')
    fig.axes[0].set_ylabel('note (MIDI code')
    plt.savefig('dnn_3_layers.png')
    plt.show()

    fig = plt.figure(figsize=(20, 5))
    plt.imshow(y.T, aspect='auto')
    plt.gca().invert_yaxis()
    fig.axes[0].set_xlabel('window')
    fig.axes[0].set_ylabel('note (MIDI code')
    plt.savefig('ground_truth.png')
    plt.show()


if __name__ == '__main__':

    X_all, y_all = None, None
    for i in range(1, 7):
        X = wav2cqt_spec('alb_esp{0}.wav'.format(i))
        times = librosa.frames_to_time(np.arange(X.shape[0]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
        y = midi2labels('alb_esp{0}.mid'.format(i), times)
        print(X.shape, y.shape)

        if i == 1:
            X_all, y_all = X, y
            predict_load(X, y)
            exit(0)
        else:
            X_all, y_all = np.concatenate((X_all, X)), np.concatenate((y_all, y))

    size = X_all.shape[0]
    half_size, third_size = size // 2, size // 2 + size // 4

    X_train, y_train = X_all[:half_size], y_all[:half_size]
    X_val, y_val = X_all[half_size:third_size], y_all[half_size:third_size]
    X_test, y_test = X_all[third_size:], y_all[third_size:]

    print('Train: ', len(X_train))
    print(X_train.shape, y_train.shape)
    print('Validation: ', len(X_val))
    print('Test: ', len(X_test))

    dnn(X_train, y_train, X_val, y_val, X_test, y_test)

