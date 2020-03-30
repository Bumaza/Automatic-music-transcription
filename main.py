from os import listdir, path
from utils.audio_analysis import *
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
from keras.optimizers import SGD
from keras import backend as K

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
    pool2 = MaxPooling2D(pool_size=(1,3))(do2)

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

    #inputnp = wav2inputnp(audio_fn, spec_type=spec_type, bin_multiple=bin_multiple)
    #times = librosa.frames_to_time(np.arange(inputnp.shape[0]), sr=sr, hop_length=hop_length)

    #piano_roll = pm.get_piano_roll(fs=sr, times=times)[MIN_MIDI_TONE:MAX_MIDI_TONE + 1].T
    #piano_roll[piano_roll > 0] = 1


    #plot_cqt(test_wav, path.join(SPECS_DIR, 'chopin.jpg'))

    #librosa.display.specshow(librosa.amplitude_to_db(d, ref = np.max),y_axis = 'log', x_axis = 'time')

    # librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max), sr=sr)
    # plt.show()
    # #TODO
    #
    # #TODO crop image
    # im = Image.open(path.join(SPECS_DIR, 'chopin.jpg'))
    # im = im.crop((14, 13, 594, 301))
    # im.show()

    # for i in range(60, 83):
    #     l = len(roll[i])
    #     r = roll[i][l//3:l//3+10]
    #     r[r > 0] = 1
    #     print(i, ": ", r)


if __name__ == '__main__':
    #test_mid2wav('new_song.mid')
    #test_wav2specs2wav()
    #make_wav_and_spectogram_files()
    #test_piano_roll('chpn-p18.mid')
    #test_piano_roll('alb_esp1.mid')]

    X = wav2cqt_spec('alb_esp1.wav')
    times = librosa.frames_to_time(np.arange(X.shape[0]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    y = midi2labels('alb_esp1.mid', times)

    print(X.shape)
    print(y.shape)

    model = baseline_model()
    model.compile(loss='binary_crossentropy', optimizer=SGD(momentum=0.9))
    print(model.summary())
    result = model.fit(X, y, epochs=3, validation_data=(X, y))
    print(result.history)



