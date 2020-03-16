import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from subprocess import call
from utils.app_setup import *


def midi2wav(midi_file, output):
    """ Convert midi file to wav
    :param midi_file:
    :param output:
    :return:
    """
    command = ['fluidsynth', '-a', 'alsa', '-F', output, SOUNDFONT, midi_file]
    return call(command)


def wav2spec(y):
    D = stft(y)
    S = amp_to_db(np.abs(D)) - REF_LEVEL_DB
    S, D = normalize(S), np.angle(D)
    S, D = S.T, D.T  # to make [time, freq]
    return S, D


def spec2wav(self, spectrogram, phase):
    spectrogram, phase = spectrogram.T, phase.T
    # used during inference only
    # spectrogram: enhanced output
    # phase: use noisy input's phase, so no GLA is required
    S = self.db_to_amp(self.denormalize(spectrogram) + self.hp.audio.ref_level_db)
    return self.istft(S, phase)


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


def stft(y):
    """ Compute STFT for audio
     :param self:
     :param y:
     :return:
     """
    return librosa.stft(y=y, n_fft=FFT_SIZE,
                        hop_length=HOP_LENGTH,
                        win_length=WIN_LENGTH)


def istft(mag, phase):
    """ Compute inverse STFT
    :param self:
    :param mag:
    :param phase:
    :return:
    """
    stft_matrix = mag * np.exp(1j * phase)
    return librosa.istft(stft_matrix,
                         hop_length=HOP_LENGTH,
                         win_length=WIN_LENGTH)


def amp_to_db(self, x):
    return 20.0 * np.log10(np.maximum(1e-5, x))


def db_to_amp(self, x):
    return np.power(10.0, x * 0.05)


def normalize(self, S):
    return np.clip(S / -MIN_LEVEL_DB, -1.0, 0.0) + 1.0


def denormalize(self, S):
    return (np.clip(S, 0.0, 1.0) - 1.0) * -MIN_LEVEL_DB