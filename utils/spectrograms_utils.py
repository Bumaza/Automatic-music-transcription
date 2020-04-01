import librosa
import librosa.display
import os
import pretty_midi
import matplotlib.pyplot as plt
from utils.audio_analysis import *
from utils.midi_utils import *
from utils.app_setup import *


def plot_stft(file, show_roll=False):
    y, sr = librosa.load(os.path.join(WAV_DIR, file))
    print('SAMPLE RATE: ', sr)
    D = librosa.stft(y)
    print(D.shape)
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                             y_axis='log', x_axis='time', sr=sr)
    plt.title('Power spectrum ' + file)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

    if show_roll:
        times = librosa.frames_to_time(np.arange(D.shape[-1]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
        showroll(times)


def plot_cqt(file, hop_length=512, bins=1, show_roll=False):
    y, sr = librosa.load(os.path.join(WAV_DIR, file))
    print('SAMPLE RATE: ', sr)
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, bins_per_octave=12*bins, n_bins=NOTE_RANGE*bins))
    print(C.shape)
    librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                             y_axis='cqt_note', x_axis='time', sr=sr, hop_length=hop_length, bins_per_octave=12*bins)
    plt.title('Constant-Q power spectrum ' + file)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

    if show_roll:
        times = librosa.frames_to_time(np.arange(C.shape[-1]), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
        showroll(times)


def showroll(file, times=None):
    pm = pretty_midi.PrettyMIDI(os.path.join(MIDI_DIR, file.replace('.wav', '.mid')))
    piano_roll = pm.get_piano_roll(fs=100, times=times)[MIN_MIDI_TONE:MAX_MIDI_TONE + 1].T
    piano_roll[piano_roll > 0] = 1
    print(piano_roll.shape)
    plt.imshow(piano_roll[:100, ])
    plt.show()
    print(piano_roll[20, ])