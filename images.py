from os import listdir, path
from utils.audio_analysis import *
from utils.midi_utils import *
from utils.app_setup import *
from PIL import Image
import librosa
import librosa.display
import os
import pretty_midi
import matplotlib.pyplot as plt

np.random.seed(400)

def stft(file):
    y, sr = librosa.load(os.path.join(WAV_DIR, file))
    print('SAMPLE RATE: ', sr)
    D = librosa.stft(y)
    print(D.shape)
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                             y_axis='log', x_axis='time', sr=sr)
    plt.title('short-time Fourier transform of ' + file)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    #plt.savefig('stft_{0}.png'.format(file.split('.')[0]))

    plt.show()


    # times = librosa.frames_to_time(np.arange(D.shape[-1]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    # showroll(times)


def plot_cqt(file, hop_length=512, bins=1):
    y, sr = librosa.load(os.path.join(WAV_DIR, file))
    print('SAMPLE RATE: ', sr)
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, bins_per_octave=12*bins, n_bins=NOTE_RANGE*bins))
    am = librosa.amplitude_to_db(C, ref=np.max)
    librosa.display.specshow(am, y_axis='cqt_note', x_axis='time', sr=sr, hop_length=hop_length, bins_per_octave=12*bins)
    plt.title('constant-Q transform of ' + file)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    #plt.savefig('cqt_{0}.png'.format(file.split('.')[0]))

    plt.show()


    # times = librosa.frames_to_time(np.arange(C.shape[-1]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    # showroll(times)


def showroll(times=None):
    song = 'alb_esp1.mid'
    pm = pretty_midi.PrettyMIDI(os.path.join(MIDI_DIR, song))
    piano_roll = pm.get_piano_roll(fs=100, times=times)[MIN_MIDI_TONE:MAX_MIDI_TONE + 1].T
    piano_roll[piano_roll > 0] = 1
    print(piano_roll.shape)

    fig = plt.figure(figsize=(22, 10))
    plt.imshow(piano_roll.T, aspect='auto')
    plt.title('piano roll of ' + song)
    plt.gca().invert_yaxis()
    fig.axes[0].set_xlabel('window')
    fig.axes[0].set_ylabel('note (MIDI code')
    plt.savefig('midi_{0}.png'.format(song.split('.')[0]))

    plt.show()



if __name__ == '__main__':
    #stft('alb_esp1.wav')
    #plot_cqt('alb_esp1.wav')
    #showroll()

    showroll()