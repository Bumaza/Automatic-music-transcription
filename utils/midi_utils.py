import numpy as np
from pretty_midi import PrettyMIDI, instrument_name_to_program, Note, Instrument
from utils.app_setup import *


def piano_roll2midi(piano_roll, fs=SAMPLE_RATE):

    frames, notes = piano_roll.shape
    pm = PrettyMIDI()
    instrument_program = instrument_name_to_program('Acoustic Grand Piano')
    instrument = Instrument(program=instrument_program)

    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')
    velocity_changes = np.nonzero(np.diff(piano_roll))

    print(velocity_changes)
    print(*velocity_changes[0])
    print(np.array(velocity_changes).shape)

    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    # for time, note in zip(*velocity_changes):
    #     velocity, time = piano_roll[time+1, note], time / fs
    #     if velocity > 0:
    #         if prev_velocities[note] == 0:
    #             note_on_time[note] = time
    #             prev_velocities[note] = velocity
    #     else:
    #         pm_note = Note(velocity=prev_velocities[note],
    #                        pitch=note+MIN_MIDI_TONE,
    #                        start=note_on_time[note], end=time)
    #         instrument.notes.append(pm_note)
    #
    # pm.instruments.append(instrument)
    # return pm


def create_midi(title, data, instrument_name='Acoustic Grand Piano', treshold=0.6):
    # Create a PrettyMIDI object
    song = PrettyMIDI()
    # Create an Instrument instance for a cello instrument
    instrument_program = instrument_name_to_program(instrument_name)
    instrument = Instrument(program=instrument_program)
    # Iterate over all note probabilities
    for sec in range(len(data)):
        # Iterate over all notes
        for note_number in range(NOTE_RANGE):
            if data[note_number] > treshold:
                # Create a Note instance for this note, starting at 0s and ending at .5s
                note = Note(velocity=100, pitch=note_number, start=0, end=.5)
                # Add it to our cello instrument
                instrument.notes.append(note)

    # Add the cello instrument to the PrettyMIDI object
    title.instruments.append(instrument)
    # Write out the MIDI data
    title.write('{}.mid'.format(title))