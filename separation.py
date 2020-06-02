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
from keras.layers import Conv2D, MaxPooling2D, add, LeakyReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.optimizers import SGD, Adam
from keras import backend as K
from sklearn.preprocessing import normalize
from transcription.cnn.dnn import DNN

sample_rate = 22050

def test_wav2specs2wav():
    audio, sr = librosa.load('datasets/separation/coze ja.wav')

    spectrum = librosa.stft(audio, hop_length=256, win_length=1024)

    reconstructed_audio = librosa.istft(spectrum, hop_length=256, win_length=1024)

    print(sum(audio[:len(reconstructed_audio)] - reconstructed_audio))

    librosa.output.write_wav('./datasets/separation/coze ja r.wav', reconstructed_audio, sr=sr)



def plot_mel():
    y, sr = librosa.load('datasets/data/mp3/alb_esp1.mp3')
    mel = librosa.feature.melspectrogram(y, SAMPLE_RATE, hop_length=512, fmin=librosa.midi_to_hz(MIN_MIDI_TONE),
                                         n_mels=229)
    librosa.display.specshow(librosa.power_to_db(mel, ref=np.max),
                             y_axis='mel', x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    plt.show()


def plot_cqt():
    y, sr = librosa.load('datasets/data/mp3/alb_esp1.mp3')
    D = np.abs(librosa.cqt(y, sr=SAMPLE_RATE, fmin=librosa.midi_to_hz(MIN_MIDI_TONE),
                    hop_length=HOP_LENGTH, bins_per_octave=BIN_PER_OCTAVE, n_bins=N_BINS))
    print(D.shape)
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                             y_axis='log', x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    plt.show()

def plot_stft2():
    y, sr = librosa.load('datasets/data/mp3/alb_esp1.mp3')
    D = librosa.stft(y)
    print(D.shape)
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                             y_axis='log', x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    plt.show()

    # reconstructed_audio = librosa.istft(D)
    #
    # print(sum(y[:len(reconstructed_audio)] - reconstructed_audio))
    #
    # librosa.output.write_wav('./datasets/separation/coze ja rr.wav', reconstructed_audio, sr=sr)

def separation():

    size = 1023

    input_shape = Input(shape=(size, 25, 1))

    x_1 = Conv2D(32, (3,3), padding='same', input_shape=(size, 25, 1))(input_shape)
    x_1 = LeakyReLU()(x_1)
    x_1 = Conv2D(16, (3,3), padding='same')(x_1)
    x_1 = LeakyReLU()(x_1)
    x_1 = MaxPooling2D(pool_size=(3,3))(x_1)
    x_1 = Dropout(0.25)(x_1)
    x_1 = Conv2D(64, (3,3), padding='same')(x_1)
    x_1 = LeakyReLU()(x_1)
    x_1 = Conv2D(16, (3,3), padding='same')(x_1)
    x_1 = LeakyReLU()(x_1)
    x_1 = MaxPooling2D(pool_size=(3,3))(x_1)
    x_1 = Dropout(0.25)(x_1)
    x_1 = Flatten()(x_1)
    x_1 = Dense(128)(x_1)
    x_1 = LeakyReLU()(x_1)
    x_1 = Dropout(0.5)(x_1)
    x_1 = Dense(size, activation='sigmoid')(x_1)

    model = Model(input_shape, x_1)
    print(model.summary())


if __name__ == '__main__':
    #test_wav2specs2wav()
    plot_stft2()
    #separation()
    plot_mel()
    plot_cqt()
    print(wav_to_mel('alb_esp1.mp3').shape)
