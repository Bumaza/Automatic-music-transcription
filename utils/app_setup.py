#some constant for AMT app

"""
single STFT/CQT frame (±11.6 ms = HOP_LENGTH / SAMPLE_RATE) (256 / 22050)
"""

SAMPLE_RATE = 16000
FFT_SIZE = 4096
WINDOW_SIZE = 20

BATCH_SIZE = 256

HOP_LENGTH = 512
WIN_LENGTH = 1024
MIN_LEVEL_DB = -100.0
REF_LEVEL_DB = 20.0

DEFAULT_BPM = 120

IMG_WIDTH = 224
IMG_HEIGHT = 224
NUM_CHANNELS = 3

INIT_LR = 0.01

LSTM_SAMPLE_SIZE = 100

MIN_MIDI_TONE = 21
MAX_MIDI_TONE = 108
NOTE_RANGE = MAX_MIDI_TONE - MIN_MIDI_TONE + 1
ALL_MIDI_TONES = 128

BIN_MULTIPLE = 4
BIN_PER_OCTAVE = 12 * BIN_MULTIPLE
N_BINS = NOTE_RANGE * BIN_MULTIPLE # 252 for DNN z netu

INPUT_SHAPE = (WINDOW_SIZE, NOTE_RANGE * BIN_MULTIPLE)
INPUT_SHAPE_CHANNEL = (WINDOW_SIZE, NOTE_RANGE * BIN_MULTIPLE, 1)

FOLK_DEBUG = False
MIDI_DIR = './datasets/data/midi/'
WAV_DIR = './datasets/data/mp3/'
if FOLK_DEBUG:
    MIDI_DIR = './datasets/folk/'
    WAV_DIR = './datasets/folk/'


SPECS_DIR = './datasets/data/spectrograms/'

MODELS_DIR = './models/'

SOUNDFONT = "./utils/FluidR3_GM2-2.SF2"
