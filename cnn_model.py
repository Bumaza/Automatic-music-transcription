from os import listdir, path
from utils.audio_analysis import *
from utils.midi_utils import *
from utils.app_setup import *
from PIL import Image
import librosa
import os
import pretty_midi
import matplotlib.pyplot as plt
from transcription.cnn.res_net import ResNet
from keras.callbacks import Callback
from keras import metrics
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, add
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.optimizers import SGD, Adam
from keras import backend as K
from sklearn.preprocessing import normalize
from transcription.cnn.dnn import DNN
from transcription.cnn.lstm import LSTM


def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def baseline_model():
    inputs = Input(shape=(WINDOW_SIZE, N_BINS))
    reshape = Reshape((WINDOW_SIZE, N_BINS, 1))(inputs)

    #normal convnet layer (have to do one initially to get 64 channels)
    conv1 = Conv2D(50, (5, 25), activation='tanh')(reshape)
    do1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling2D(pool_size=(1,3))(do1)

    conv2 = Conv2D(50,(3, 5),activation='tanh')(pool1)
    do2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling2D(pool_size=(1,3))(do2)

    flattened = Flatten()(pool2)
    fc1 = Dense(1000, activation='sigmoid')(flattened)
    do3 = Dropout(0.5)(fc1)

    fc2 = Dense(200, activation='sigmoid')(do3)
    do4 = Dropout(0.5)(fc2)
    outputs = Dense(NOTE_RANGE, activation='sigmoid')(do4)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_model():
    # input and reshape
    inputs = Input(shape=(WINDOW_SIZE, N_BINS))
    reshape = Reshape((WINDOW_SIZE, N_BINS, 1))(inputs)

    # normal convnet layer (have to do one initially to get 64 channels)
    conv = Conv2D(64, (1, BIN_MULTIPLE * NOTE_RANGE), padding="same", activation='relu')(reshape)
    pool = MaxPooling2D(pool_size=(1, 2))(conv)

    for i in range(int(np.log2(BIN_MULTIPLE)) - 1):

        # residual block
        bn = BatchNormalization()(pool)
        re = Activation('relu')(bn)
        freq_range = int((BIN_MULTIPLE / (2 ** (i + 1))) * NOTE_RANGE)

        conv = Conv2D(64, (1, freq_range), padding="same", activation='relu')(re)

        # add and downsample
        ad = add([pool, conv])
        pool = MaxPooling2D(pool_size=(1, 2))(ad)

    flattened = Flatten()(pool)
    fc = Dense(1024, activation='relu')(flattened)
    do = Dropout(0.5)(fc)
    fc = Dense(512, activation='relu')(do)
    do = Dropout(0.5)(fc)
    outputs = Dense(NOTE_RANGE, activation='sigmoid')(do)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def train(model, X_train, y_train, X_val, y_val, X_test):

    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=INIT_LR, momentum=0.9), metrics=[get_f1, 'accuracy'])
    print(model.summary())

    checkpoint = ModelCheckpoint(filepath='model.hdf5', verbose=1, save_best_only=False)
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')
    callbacks = [checkpoint, earlystop]

    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=1000,
                        validation_data=(X_val, y_val), verbose=1, callbacks=callbacks)

    print(history.history)

    y_pred = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    plot_predict(postprocess(y_pred))


def postprocess(y_pred):

    y_pred = np.array(y_pred).round()
    y_pred[y_pred > 1] = 1

    changes = 0

    for note in range(y_pred.shape[1]):
        for frame in range(2, y_pred.shape[0] - 3):

            if list(y_pred[frame - 1:frame + 3, note]) == [1.0, 0.0, 0.0, 1.0]:
                y_pred[frame, note], y_pred[frame + 1, note] = 1, 1
                changes += 1

            if list(y_pred[frame - 2:frame + 4, note]) == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]:
                y_pred[frame, note], y_pred[frame + 1, note] = 0, 0
                changes += 1

            if list(y_pred[frame - 1:frame + 3, note]) == [0.0, 1.0, 0.0, 0.0]:
                y_pred[frame, note] = 0
                changes += 1

            if list(y_pred[frame - 1:frame + 3, note]) == [1.0, 0.0, 1.0, 1.0]:
                y_pred[frame, note] = 1
                changes += 1

    print('Total changes: {0}'.format(changes))
    return y_pred


def plot_predict(y):
    fig = plt.figure(figsize=(20, 5))
    plt.title('Predicted')
    plt.imshow(y.T, aspect='auto')
    plt.gca().invert_yaxis()
    fig.axes[0].set_xlabel('window')
    fig.axes[0].set_ylabel('note (MIDI code')
    plt.show()


def folk_dataset(just_test=False):
    X_polovnicek = wav2cqt_spec('polovnicek.mp3', True)
    times = librosa.frames_to_time(np.arange(X_polovnicek.shape[0]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    y_polovnicek = midi2labels('polovnicek.MID', times)
    #
    X_jedna = wav2cqt_spec('jedna.mp3', True)
    times = librosa.frames_to_time(np.arange(X_jedna.shape[0]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    y_jedna = midi2labels('jedna.MID', times)
    #
    X_kohutik = wav2cqt_spec('kohutik.mp3', True)
    times = librosa.frames_to_time(np.arange(X_kohutik.shape[0]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    y_kohutik = midi2labels('kohutik.MID', times)

    X_hora = wav2cqt_spec('hora.mp3', True)
    times = librosa.frames_to_time(np.arange(X_hora.shape[0]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    y_hore = midi2labels('hora.mid', times)

    X_train, y_train = np.concatenate((X_kohutik, X_polovnicek)), np.concatenate((y_kohutik, y_polovnicek))

    print(X_train.shape, y_train.shape)

    if just_test:
        model = load_model('model.hdf5', custom_objects={"get_f1": get_f1})
        y_pred = model.predict(X_hora, batch_size=BATCH_SIZE, verbose=1)
        plot_predict(postprocess(y_pred))
        return

    train(resnet_model(), X_train, y_train, X_jedna, y_jedna, X_hora)


def piano_dataset():
    pass


def main():
    folk_dataset()


if __name__ == '__main__':
    main()