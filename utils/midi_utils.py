import numpy as np
from pretty_midi import PrettyMIDI, instrument_name_to_program, Note, Instrument
from utils.app_setup import *


def piano_roll2midi(piano_roll):
    pass


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