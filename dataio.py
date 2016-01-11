from numpy import vstack
import csv

"""
This file contains functions that read the signal and signal information from file.
It also contains functions that modify the signal data in file.
"""

def load_signals(e, signal_indices):
    """Load and return signal from file
    :param e: EdfReader object.
    :param signal_indices: array with the indices of the signals from which to extract features.
    """
    # Store signals in RAM
    eeg1 	= e.readSignal(signal_indices[1])
    # eeg2 	= e.readSignal(1)
    eog 	= e.readSignal(signal_indices[0])
    resp 	= e.readSignal(signal_indices[2])
    # emg 	= e.readSignal(4)
    # temp 	= e.readSignal(5)
    return eog, eeg1, resp

def get_signal_frequencies(e, signal_indices):
    """Return an array containing the frequencies for the signals with indices given in @signal_indices
    :param e: the EdfReader object.
    :param signal_indices: the indices of the signals to retrieve frequencies from.
    :return: an array with the frequencies of the given signals.
    """
    all_freqs = e.getSignalFreqs()
    return [ int(all_freqs[i]) for i in signal_indices ]

def get_signals(e, signal_indices):
    """Load an return signals from file
    :param e: the EdfReader object.
    :param signal_indices: the indices of the signals to retrieve data from.
    : return: an array of signal arrays.
    """
    # :return data: a numpy array with dimensions (n_signals, n_points), where each row contains the data of a signal in @signal_indices.

    return [e.readSignal(idx) for idx in signal_indices]

def get_signal_freqs(e):
    """Return signal frequences
    :param e: the EdfReader object.
    :return: array of frequencies for the different signals in the EDF file.
    """
    freqs = e.getSignalFreqs()
    return [freqs[i] for i in range(0, 6)]

def reduce_label_dimensions(raw, reduced):
    """Reduce dimensionality of possible stages as follows:
        1, 2, 3 and 4 -> NREM (N)
        0, 6 -> Wake / Movement (W)
        5 -> REM (R)
        9 -> Unscored: ignore
        :param raw: filename of raw csv file
        :param reduced: filename of reduced csv file
    """
    # infile = '/Users/mimoun/dsp/data/200002/shhs1-200002-staging.csv'
    f = open(raw, 'r')
    # outfile = '/Users/mimoun/dsp/data/200002/shhs1-200002-staging-reduced.csv'
    o = open(reduced, 'w+')

    reader = csv.reader(f)
    writer = csv.writer(o)

    for row in reader:
        # Write row: [ epoch, reduced_stage ]
        # rs: reduced stage
        rs = ""
        if row[1] == "1" or row[1] == "2" or row[1] == "3" or row[1] == "4":
            rs = "N"
        if row[1] == "0" or row[1] == "6":
            rs = "W"
        if row[1] == "5":
            rs = "R"
        if row[1] == "9":
            continue

        writer.writerow([row[0], rs])

def get_stages_array(reduced_label_file):
    """Returns an array of sleep stages given the filename.
    :param reduced_label_file: the filename of the csv file with the reduced labels.
    :return stages: an array of length (num_epoch) with the labels of the different sleep epochs (W, N, R)
    """
    f = open(reduced_label_file, 'r')
    reader = csv.reader(f)

    reader.next()
    stages = []
    for row in reader:
        stages.append(row[1])

    return stages