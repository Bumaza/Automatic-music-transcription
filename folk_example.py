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


def test_mid2wav(filename='jedna.MID'):
    test_mid = './datasets/folk/'+filename
    test_wav = './datasets/folk/'+filename.replace('.mid', '_synth.wav')
    midi2wav(test_mid, test_wav)


def showroll(name='kohutik.MID', times=None):
    song = name
    pm = pretty_midi.PrettyMIDI(os.path.join('./datasets/midi/', song))

    piano_roll = pm.get_piano_roll(fs=100, times=times)[MIN_MIDI_TONE:MAX_MIDI_TONE + 1].T
    piano_roll[piano_roll > 0] = 1

    fig = plt.figure(figsize=(22, 10))
    plt.imshow(piano_roll.T, aspect='auto')
    plt.title('piano roll of ' + song)
    plt.gca().invert_yaxis()
    fig.axes[0].set_xlabel('window')
    fig.axes[0].set_ylabel('note (MIDI code')
    #plt.savefig('midi_{0}.png'.format(song.split('.')[0]))
    print(piano_roll.shape, len(piano_roll), len(piano_roll[-1]))
    print(piano_roll)
    plt.show()


def plot_cqt(file, hop_length=512, bins=1, roll=False):
    y, sr = librosa.load(os.path.join('./datasets/folk/', file))
    print('SAMPLE RATE: ', sr)
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, bins_per_octave=12*bins, n_bins=NOTE_RANGE*bins))
    am = librosa.amplitude_to_db(C, ref=np.max)
    librosa.display.specshow(am, y_axis='cqt_note', x_axis='time', sr=sr, hop_length=hop_length, bins_per_octave=12*bins)
    plt.title('constant-Q transform of ' + file)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    #plt.savefig('cqt_{0}.png'.format(file.split('.')[0]))

    plt.show()

    if roll:
        times = librosa.frames_to_time(np.arange(C.shape[-1]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
        showroll(times)


def test_model_on_folk():

    # X_polovnicek = wav2cqt_spec('polovnicek.wav')
    # times = librosa.frames_to_time(np.arange(X_polovnicek.shape[0]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    # y_polovnicek = midi2labels('polovnicek.MID', times)
    # #
    # X_jedna = wav2cqt_spec('jedna.mp3')
    # times = librosa.frames_to_time(np.arange(X_jedna.shape[0]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    # y_jedna = midi2labels('jedna.MID', times)
    # #
    # X_kohutik = wav2cqt_spec('kohutik.wav')
    # times = librosa.frames_to_time(np.arange(X_kohutik.shape[0]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    # y_kohutik = midi2labels('kohutik.MID', times)
    #
    # X_marienka = wav2cqt_spec('marienka.mp3')
    # times = librosa.frames_to_time(np.arange(X_marienka.shape[0]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    # y_marienka = midi2labels('marienka.mid', times)
    #
    # X_hora = wav2cqt_spec('hora.mp3')
    # times = librosa.frames_to_time(np.arange(X_hora.shape[0]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    # y_hore = midi2labels('hora.mid', times)
    #
    # X_onvo = wav2cqt_spec('onvo.mp3')
    # times = librosa.frames_to_time(np.arange(X_marienka.shape[0]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    # y_onvo = midi2labels('onvo.mid', times)

    # model = load_model('DNN_mp3_piano.hdf5')
    # dnn = DNN(3, 256)
    # dnn.set_model(model)
    # dnn.summary()
    #
    #
    #
    #
    # dnn.predict(X_kohutik, y_kohutik)
    #dnn.predict(X_polovnicek, y_polovnicek)
    # X = wav2cqt_spec('MAPS_MUS-alb_esp2_AkPnCGdD.flac')
    # dnn.predict(X)

    # X = wav2cqt_spec('alb_esp{0}.wav'.format(1))
    # times = librosa.frames_to_time(np.arange(X.shape[0]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    # y = midi2labels('alb_esp{0}.mid'.format(1), times)
    #
    # dnn.predict(X, y)

    #exit(0)



    X_all, y_all = None, None


    for i in range(1, 7):
        X = wav2cqt_spec('alb_esp{0}.mp3'.format(i))
        times = librosa.frames_to_time(np.arange(X.shape[0]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
        y = midi2labels('alb_esp{0}.mid'.format(i), times)
        print(X.shape, y.shape)

        if i == 1:
            X_all, y_all = X, y
        else:
            X_all, y_all = np.concatenate((X_all, X)), np.concatenate((y_all, y))

    # wavs = [x for x in listdir(WAV_DIR) if x.endswith('.mp3') and 'format0' not in x]
    # np.random.seed()
    # np.random.shuffle(wavs)
    #
    # i, length = 1, len(wavs)
    # X_all, y_all = None, None
    # for wav in wavs:
    #     try:
    #         X = wav2cqt_spec(wav)
    #         times = librosa.frames_to_time(np.arange(X.shape[0]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    #         y = midi2labels('{0}.mid'.format(wav.split('.')[0]), times)
    #
    #         if X_all is None or y_all is None:
    #             X_all, y_all = X, y
    #         elif X.shape[0] == y.shape[0]:
    #             X_all, y_all = np.concatenate((X_all, X)), np.concatenate((y_all, y))
    #
    #         print('{0}/{1} {2} {3}.mid'.format(i, length, wav, wav.split('.')[0]), X.shape, '/', X_all.shape, y.shape,
    #               '/', y_all.shape)
    #         i += 1
    #
    #         if i >= 20:
    #             break
    #
    #     except FileNotFoundError as err:
    #         print(err)
    #     except Exception as err:
    #         print(err)




    min_all, max_all = X_all.min(axis=0), X_all.max(axis=0)
    X_all = (X_all - min_all) / (max_all - min_all)

    size = X_all.shape[0]
    half_size, third_size = size // 2, size // 2 + size // 4

    X_train, y_train = X_all[:half_size], y_all[:half_size]
    X_val, y_val = X_all[half_size:third_size], y_all[half_size:third_size]
    X_test, y_test = X_all[third_size:], y_all[third_size:]



    # dnn = DNN(256, 3)
    # dnn.create()
    # dnn.train(X_train, y_train, X_val, y_val)
    # dnn.predict(X_test, y_test)

    X_train = np.array([X_train[i:i + LSTM_SAMPLE_SIZE, :] for i in range(0, len(X_train) - LSTM_SAMPLE_SIZE + 1, LSTM_SAMPLE_SIZE)])
    y_train = np.array([y_train[i:i + LSTM_SAMPLE_SIZE, :] for i in range(0, len(y_train) - LSTM_SAMPLE_SIZE + 1, LSTM_SAMPLE_SIZE)])

    X_val = np.array([X_val[i:i + LSTM_SAMPLE_SIZE, :] for i in range(0, len(X_val) - LSTM_SAMPLE_SIZE + 1, LSTM_SAMPLE_SIZE)])
    y_val = np.array([y_val[i:i + LSTM_SAMPLE_SIZE, :] for i in range(0, len(y_val) - LSTM_SAMPLE_SIZE + 1, LSTM_SAMPLE_SIZE)])

    X_test = np.array([X_test[i:i + LSTM_SAMPLE_SIZE, :] for i in range(0, len(X_test) - LSTM_SAMPLE_SIZE + 1, LSTM_SAMPLE_SIZE)])
    y_val = np.array([y_test[i:i + LSTM_SAMPLE_SIZE, :] for i in range(0, len(y_test) - LSTM_SAMPLE_SIZE + 1, LSTM_SAMPLE_SIZE)])

    try:


        lstm = LSTM(256, 3)
        lstm.create()
        lstm.summary()
        lstm.train(X_train, y_train, X_val, y_val)
        lstm.predict(X_test, y_test)
    except Exception as ex:
        print(ex)

    # train, val, test = ['polovnicek.mp3', 'marienka.mp3', 'hora.mp3', 'onvo.mp3'], ['jedna.mp3'], ['kohutik.mp3']
    #
    # #train, val, test = ['polovnicek.wav'], ['polovnicek.wav'], ['kohutik.wav']
    #
    # X_train, y_train = None, None
    # X_val, y_val = None, None
    # X_test, y_test = None, None
    #
    # for song in train:
    #     X = wav2cqt_spec(song)
    #     times = librosa.frames_to_time(np.arange(X.shape[0]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    #     y = midi2labels(song.replace('.mp3', '.mid'), times)
    #     if X_train is None:
    #         X_train, y_train = X, y
    #     else:
    #         X_train, y_train = np.concatenate((X_train, X)), np.concatenate((y_train, y))
    #
    # for song in val:
    #     X = wav2cqt_spec(song)
    #     times = librosa.frames_to_time(np.arange(X.shape[0]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    #     y = midi2labels(song.replace('.mp3', '.mid'), times)
    #     if X_val is None:
    #         X_val, y_val = X, y
    #     else:
    #         X_val, y_val = np.concatenate((X_val, X)), np.concatenate((y_val, y))
    #
    # for song in test:
    #     X = wav2cqt_spec(song)
    #     times = librosa.frames_to_time(np.arange(X.shape[0]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    #     y = midi2labels(song.replace('.mp3', '.mid'), times)
    #     if X_test is None:
    #         X_test, y_test = X, y
    #     else:
    #         X_test, y_test = np.concatenate((X_test, X)), np.concatenate((y_test, y))




    #
    # dnn.summary()
    #
    # X_train, y_train = np.concatenate((X_kohutik, X_jedna)), np.concatenate((y_kohutik, y_jedna))
    # X_train, y_train = np.concatenate((X_train, X_polovnicek)), np.concatenate((y_train, y_polovnicek))
    # print(X_polovnicek.shape, X_jedna.shape)
    # print(X_train.shape)
    #
    # dnn.train(X_kohutik, y_kohutik, X_kohutik, y_kohutik)
    #
    # dnn.predict(X_kohutik)
    # dnn.predict(X_jedna)
    # dnn.predict(X_polovnicek)
    #dnn.predict(X_hora)

    # lstm = LSTM(256, 3)
    # lstm.create()
    # lstm.summary()
    # lstm.train(X_train, y_train, X_val, y_val)
    # lstm.predict(X_test, y_test)


    #dnn.predict(X_test[:min(len(X_test), 2000)], y_test[:min(len(y_test), 2000)])
    # dnn.predict(X_kohutik, y_kohutik)
    # dnn.predict(X_polovnicek, y_polovnicek)


if __name__ == '__main__':
    #test_mid2wav('polovnicek.MID')
    #plot_cqt('jedna.mp3')
    # plot_cqt('kohutik.MP3', roll=False)
    # showroll(name='kohutik.MID')
    #plot_cqt('vychodne.mp4')
    # showroll('accordeon/Midi example.mid')
    # showroll('bass/Midi example.mid')
    # showroll('bracsa/Midi example.mid')

    #print(N_BINS)
    # lstm = LSTM(256, 3)
    # lstm.create()
    # lstm.summary()

    test_model_on_folk()


    #test_model_on_folk()
    #test_mid2wav('hora.mid')
    # test_mid2wav('marienka.mid')
    # test_mid2wav('duj duj.mid')
    # test_mid2wav('cimbal daco.mid')

