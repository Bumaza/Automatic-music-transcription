from os import listdir
from utils.audio_analysis import *
from utils.app_setup import *
import librosa

def make_wav_and_spectogram_files():

    midis = [x for x in listdir(MIDI_DIR) if x.endswith('.mid')]
    wavs = [x for x in listdir(WAV_DIR) if x .endswith('.wav')]
    specs = [x for x in listdir(SPECS_DIR) if x.endswith('.jpg')]


def test_mid2wav(filename):
    test_mid = './datasets/data/mid/'+filename
    test_wav = './datasets/data/wav/'+filename.replace('.mid', '.wav')
    midi2wav(test_mid, test_wav)


def test_wav2specs2wav():
    wav, _ = librosa.load('datasets/data/wav/test.wav', sr=16000)
    mag, phase = wav2spec(wav)

    est_wav = spec2wav(mag, phase)
    librosa.output.write_wav('datasets/data/wav/rtest.wav', est_wav, sr=16000)


if __name__ == '__main__':
    test_mid2wav('new_song.mid')
    #test_wav2specs2wav()
