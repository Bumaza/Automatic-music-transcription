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
from numpy import newaxis



def test():
    mixed_wav, sr = librosa.load('./datasets/separation/mix/mix7.wav')
    mag, phase = wav2spec(mixed_wav)
    print(mag.shape)
    WINDOW_SIZE = 25
    SIZE = mag.shape[0]

    X_test = np.array([mag[i:i + WINDOW_SIZE, :] for i in range(0, SIZE - WINDOW_SIZE + 1)])[..., newaxis]
    model = load_model('separation_violin.hdf5')


    mask = model.predict(X_test)

    l = len(mask)

    est_mag = mag[:l] * mask
    est_wav = spec2wav(est_mag, phase[:l])

    librosa.output.write_wav('./datasets/separation/tomas_7.wav', est_wav, sr=sr)

    # audio, sr = librosa.load('./datasets/separation/mix.mp3')
    # spectrum = librosa.stft(audio, hop_length=256, win_length=1024)[:, 8000:14000]
    # amp = librosa.amplitude_to_db(spectrum, ref=np.max)
    # librosa.display.specshow(amp, y_axis='log', x_axis='time', sr=sr)
    # print(amp.shape)
    # print(amp[:1])
    # plt.colorbar(format='%+2.0f dB')
    # plt.tight_layout()
    # plt.show()
    #
    # WINDOW_SIZE, SIZE = 25, spectrum.shape[1]
    #
    # windows_mix = np.array([spectrum[:, i:i + WINDOW_SIZE] for i in range(0, SIZE - WINDOW_SIZE + 1)])[..., newaxis]
    # print(windows_mix.shape)
    #
    # model = load_model('separation.hdf5')
    # bracsa = model.predict(windows_mix, verbose=1)
    #
    # print(bracsa.shape)
    # print(bracsa[:1])
    #
    # reconstructed_audio = librosa.istft(bracsa, hop_length=256, win_length=1024)
    #
    # librosa.output.write_wav('./datasets/separation/bracsa_rev.wav', reconstructed_audio, sr=sr)

    plot_rev()


def preprocess():
    audio, sr = librosa.load('./datasets/separation/skuska2_basa.mp3')
    k, prev, step = 0, 0, len(audio) // 12
    for i in range(step, len(audio), step):
        librosa.output.write_wav('./datasets/separation/bass/basa{0}.wav'.format(k), audio[prev:i], sr=sr)
        prev, k = i, k+1

def plot_rev():
    audio, sr = librosa.load('./datasets/separation/bracsa_sihelske2.wav')

    #spectrum = librosa.stft(audio, hop_length=256, win_length=1024)
    mag, phase = wav2spec(audio)
    #db = librosa.amplitude_to_db(spectrum, ref=np.max)
    mag[mag > 1] = 1
    librosa.display.specshow(mag.T, y_axis='log', x_axis='time', sr=sr)

    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    #test()
    preprocess()
    #plot_rev()