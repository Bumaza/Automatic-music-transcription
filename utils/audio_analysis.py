import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from subprocess import call
from utils.app_setup import *

def midi_to_wav(midi_file, output):
    """ Convert midi file to wav
    :param midi_file:
    :param output:
    :return:
    """
    command = ['fluidsynth', '-a', 'alsa', '-F', output, SOUNDFONT, midi_file]
    return call(command)


def wav_to_spectrogram(wav_file, output):
    """ Convert wav file to spectogram
    :param wav_file:
    :param output:
    :return:
    """
    plot_cqt(wav_file, output)


def plot_cqt(song, path):
    """ Save cqt plot of audio signal
    :param song:
    :param path:
    :return:
    """
    plt.figure(figsize=(7.5, 3.75))
    y, sr = librosa.load(song)
    C = librosa.cqt(y, sr=sr)
    librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max), sr=sr)
    plt.axis('off')
    plt.savefig(path, bbox_inches="tight")
    plt.close('all')
