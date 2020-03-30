import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
import os
from subprocess import call
from utils.app_setup import *


def midi2wav(midi_file, output):
    """ Convert midi file to wav
    before run make sure you have installed fluidsynth
    (type 'brew install fluidsynth' to your console)
    :param midi_file:
    :param output:
    :return:
    """
    command = ['fluidsynth', '-a', 'alsa', '-F', output, SOUNDFONT, midi_file]
    return call(command)


def wav2spec(y):
    """ Transform wav file amplitude to
    time-frequency domain using stft algorithm
    :param y:
    :return:
    """
    D = stft(y)
    S = amp_to_db(np.abs(D)) - REF_LEVEL_DB
    S, D = normalize(S), np.angle(D)
    S, D = S.T, D.T  # to make [time, freq]
    return S, D


def spec2wav(spectrogram, phase):
    """

    :param spectrogram:
    :param phase:
    :return:
    """
    spectrogram, phase = spectrogram.T, phase.T
    # used during inference only
    # spectrogram: enhanced output
    # phase: use noisy input's phase, so no GLA is required
    S = db_to_amp(denormalize(spectrogram) + REF_LEVEL_DB)
    return istft(S, phase)


def wav2cqt_spec(wav_file):
    """ Convert wav file to cqT spectogram
    :param wav_file:
    :return:
    """
    y, sr = librosa.load(os.path.join(WAV_DIR, wav_file))
    C = librosa.cqt(y, sr=sr, fmin=librosa.midi_to_hz(MIN_MIDI_TONE),
                    hop_length=HOP_LENGTH, bins_per_octave=BIN_PER_OCTAVE, n_bins=N_BINS).T
    C = np.abs(C)
    minDB = np.min(C)

    C = np.pad(C, ((WINDOW_SIZE//2, WINDOW_SIZE//2), (0, 0)), 'constant', constant_values=minDB)
    windows = [ C[i:i+WINDOW_SIZE,:] for i in range(C.shape[0] - WINDOW_SIZE + 1)]
    return np.array(windows)


def midi2labels(midi_file, times):
    """ Convert midi file to a piano roll
    :param midi_file:
    :param times:
    :return:
    """
    pm = pretty_midi.PrettyMIDI(os.path.join(MIDI_DIR, midi_file))
    piano_roll = pm.get_piano_roll(fs=SAMPLE_RATE, times=times)[MIN_MIDI_TONE:MAX_MIDI_TONE + 1].T
    piano_roll[piano_roll > 0] = 1
    return piano_roll


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
     :param y:
     :return:
     """
    return librosa.stft(y=y, n_fft=FFT_SIZE,
                        hop_length=HOP_LENGTH,
                        win_length=WIN_LENGTH)


def istft(mag, phase):
    """ Compute inverse STFT
    :param mag:
    :param phase:
    :return:
    """
    stft_matrix = mag * np.exp(1j * phase)
    return librosa.istft(stft_matrix,
                         hop_length=HOP_LENGTH,
                         win_length=WIN_LENGTH)


def amp_to_db(x):
    return 20.0 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
    return np.power(10.0, x * 0.05)


def normalize(S):
    return np.clip(S / -MIN_LEVEL_DB, -1.0, 0.0) + 1.0


def denormalize(S):
    return (np.clip(S, 0.0, 1.0) - 1.0) * -MIN_LEVEL_DB