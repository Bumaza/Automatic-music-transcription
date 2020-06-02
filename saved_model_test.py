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


def main():

    X, y = None, None

    if FOLK_DEBUG:
        X = wav2cqt_spec('polovnicek.MP3')
        times = librosa.frames_to_time(np.arange(X.shape[0]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
        y = midi2labels('polovnicek.MID', times)
    else:
        X = wav2cqt_spec('alb_esp{0}.wav'.format(1))
        times = librosa.frames_to_time(np.arange(X.shape[0]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
        y = midi2labels('alb_esp{0}.mid'.format(1), times)

    model = load_model('lstm3.hdf5')

    dnn = DNN(3, 256)
    dnn.set_model(model)

    dnn.predict(X, y)


if __name__ == '__main__':
    main()