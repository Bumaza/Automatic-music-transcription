from utils.audio_analysis import *
from utils.midi_utils import *
from utils.app_setup import *
from os import listdir, path


def sequential_preprocess():

    wavs = [x for x in listdir(WAV_DIR) if x.endswith('.wav') and 'format0' not in x]
    np.random.shuffle(wavs)

    i, length = 1, len(wavs)
    X_all, y_all = None, None
    for wav in wavs:
        try:
            X = wav2cqt_spec(wav)
            times = librosa.frames_to_time(np.arange(X.shape[0]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
            y = midi2labels('{0}.mid'.format(wav.split('.')[0]), times)

            if X_all is None or y_all is None:
                X_all, y_all = X, y
            elif X.shape[0] == y.shape[0]:
                X_all, y_all = np.concatenate((X_all, X)), np.concatenate((y_all, y))

            print('{0}/{1} {2} {3}.mid'.format(i, length, wav, wav.split('.')[0]), X.shape, '/', X_all.shape, y.shape, '/', y_all.shape)
            i += 1

        except FileNotFoundError as err:
            print(err)
        except Exception as err:
            print(err)

    print(X_all.shape, y_all.shape)

    np.save('input_sequential_data2.npy', X_all) # ~10GB
    np.save('output_sequential_data2.npy', y_all)# ~2GB

    # X_all, y_all = np.load('input_sequential_data.npy'), np.load('output_sequential_data.npy')
    # print(X_all.shape, y_all.shape)


if __name__ == '__main__':

    # total time of preprocessing 33GB wav files: ~5h
    sequential_preprocess()