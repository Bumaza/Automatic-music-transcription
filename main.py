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

def make_wav_and_spectogram_files():

    midis = [x for x in listdir(MIDI_DIR) if x.endswith('.mid')]
    wavs = [x for x in listdir(WAV_DIR) if x .endswith('.wav')]
    specs = [x for x in listdir(SPECS_DIR) if x.endswith('.jpg')]

    for midi in midis:
       test_mid2wav(midi)


def test_mid2wav(filename):
    test_mid = MIDI_DIR+filename
    test_wav = WAV_DIR+filename.replace('.mid', '.wav')
    midi2wav(test_mid, test_wav)


def test_wav2specs2wav():
    wav, _ = librosa.load('./datasets/data/wav/test.wav', sr=16000)
    mag, phase = wav2spec(wav)

    est_wav = spec2wav(mag, phase)
    librosa.output.write_wav('./datasets/data/wav/rtest.wav', est_wav, sr=16000)


def baseline_model():
    inputs = Input(shape=INPUT_SHAPE)
    reshape = Reshape(INPUT_SHAPE_CHANNEL)(inputs)

    #normal convnet layer (have to do one initially to get 64 channels)
    conv1 = Conv2D(50,(5,25),activation='tanh')(reshape)
    do1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling2D(pool_size=(1,3))(do1)

    conv2 = Conv2D(50,(3,5),activation='tanh')(pool1)
    do2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling2D(pool_size=(1, 3))(do2)

    flattened = Flatten()(pool2)
    fc1 = Dense(1000, activation='sigmoid')(flattened)
    do3 = Dropout(0.5)(fc1)

    fc2 = Dense(200, activation='sigmoid')(do3)
    do4 = Dropout(0.5)(fc2)
    outputs = Dense(NOTE_RANGE, activation='sigmoid')(do4)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def test_piano_roll(filename):
    test_mid = path.join(MIDI_DIR, filename)
    test_wav = WAV_DIR + filename.replace('.mid', '.wav')
    pm = pretty_midi.PrettyMIDI(test_mid)
    print(pm.instruments, len(pm.instruments))
    roll = pm.instruments[0].get_piano_roll()
    print(roll.shape)

    y, sr = librosa.load(test_wav)
    print("STFT")
    s, d = wav2spec(y)
    print(s.shape, d.shape)

    C = librosa.cqt(y, sr=sr)
    print("cQT")
    print(C.shape)


class linear_decay(Callback):
    '''
        decay = decay value to subtract each epoch
    '''
    def __init__(self, initial_lr,epochs):
        super(linear_decay, self).__init__()
        self.initial_lr = initial_lr
        self.decay = initial_lr/epochs

    def on_epoch_begin(self, epoch, logs={}):
        new_lr = self.initial_lr - self.decay*epoch
        print("ld: learning rate is now "+str(new_lr))
        K.set_value(self.model.optimizer.lr, new_lr)


class Threshold(Callback):
    '''
        decay = decay value to subtract each epoch
    '''
    def __init__(self, val_data):
        super(Threshold, self).__init__()
        self.val_data = val_data
        _,y = val_data
        self.othresholds = np.full(y.shape[1],0.5)

    def on_epoch_end(self, epoch, logs={}):
        #find optimal thresholds on validation data
        x,y_true = self.val_data
        y_scores = self.model.predict(x)
        self.othresholds = self.opt_thresholds(y_true,y_scores)
        y_pred = y_scores > self.othresholds
        p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_true,y_pred,average='micro')
        print("validation p,r,f,s:", p,r,f,s)

    def opt_thresholds(y_true, y_scores):
        othresholds = np.zeros(y_scores.shape[1])
        print(othresholds.shape)
        for label, (label_scores, true_bin) in enumerate(zip(y_scores.T, y_true.T)):
            # print label
            precision, recall, thresholds = sklearn.metrics.precision_recall_curve(true_bin, label_scores)
            max_f1 = 0
            max_f1_threshold = .5
            for r, p, t in zip(recall, precision, thresholds):
                if p + r == 0: continue
                if (2 * p * r) / (p + r) > max_f1:
                    max_f1 = (2 * p * r) / (p + r)
                    max_f1_threshold = t
            # print label, ": ", max_f1_threshold, "=>", max_f1
            othresholds[label] = max_f1_threshold
            print(othresholds)
        return othresholds


def show_spectrograms():
    pass


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def load_data_and_test_3dnn():
    # load preprocessing data
    # X_all, y_all = np.load('input_sequential_data.npy'), np.load('output_sequential_data.npy')
    # size = X_all.shape[0]
    #
    # # split data by ration 60:20:20
    # s_60p, s_20p = int(size * 0.6), int(size * 0.2)
    #
    # X_train, y_train = X_all[:s_60p], y_all[:s_60p]
    # X_val, y_val = X_all[s_60p:s_60p+s_20p], y_all[s_60p:s_60p+s_20p]
    # X_test, y_test = X_all[s_60p+s_20p:], y_all[s_60p+s_20p:]

    dnn = DNN(256, 3)
    dnn.create()

    dnn.summary()
    #dnn.train(X_train, y_train, X_val, y_val)
    #dnn.predict(X_test[:min(len(X_test, 3000))], y_test[:min(len(X_test, 3000))])



def separate_channels():
    prefix = './datasets/folk/'
    midis = [x for x in listdir(prefix) if x.lower().endswith('.mid')]
    ins = ['accordeon', 'bass', 'bracsa']

    for midi in midis:
        pm = pretty_midi.PrettyMIDI(prefix+midi)
        print(midi, len(pm.instruments))
        if len(pm.instruments) >= 3:
            print('SPLITING ', midi)
            for i in range(len(pm.instruments)):
                cpm = pretty_midi.PrettyMIDI(prefix + midi)
                cpm.instruments = [pm.instruments[i]]
                cpm.write('./datasets/midi/{0}/{1}'.format(ins[i], midi))



if __name__ == '__main__':
    #load_data_and_test_3dnn()
    #separate_channels()
    midi2wav('./datasets/folk/13.mid', './datasets/folk/test.wav')



