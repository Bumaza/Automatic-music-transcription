import pretty_midi
import matplotlib.pyplot as plt
from os import listdir
import seaborn as sns
import matplotlib.style as style
from utils.app_setup import MIN_MIDI_TONE, MAX_MIDI_TONE, NOTE_RANGE

def main():


    prefix = 'data/midi/'

    midis = [x for x in listdir(prefix) if x.endswith('.mid') and 'format0' not in x]
    tones = [0] * NOTE_RANGE

    pretty_midi.pretty_midi.MAX_TICK = 1e10

    for midi in midis:
        midi_data = pretty_midi.PrettyMIDI(prefix+midi)
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    try:
                        if note.pitch - MIN_MIDI_TONE >= 0:
                            tones[note.pitch - MIN_MIDI_TONE] += 1
                        else:
                            print(note.pitch)
                    except IndexError as ex:
                        print(note.pitch)

    plt.title('Piano MIDI dataset - Note Histogram')
    plt.ylabel('Occurences')
    plt.xlabel('Note (MIDI code)')
    barplot = plt.bar([i for i in range(MIN_MIDI_TONE, MAX_MIDI_TONE+1)], tones)

    for i in range(len(barplot)):
        barplot[i].set_color('black')
    # plt.xticks(np.arange(len(class_names)), class_names)
    # plt.yticks(np.arange(0, max(instruments_counts), max(instruments_counts) // 10))
    plt.savefig('piano_histogram.png')

    plt.show()

    m = max(tones)
    print([i + MIN_MIDI_TONE for i, j in enumerate(tones) if j == m])


if __name__ == '__main__':
    main()