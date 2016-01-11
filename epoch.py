from numpy import arange

"""
This file contains functions that interact with signal data at the epoch level.
"""

def get_epoch_data(sig, freq, ei, ne, mute):
    """This function retrieves @ne epochs worth of signal @sig (passed by reference)
    starting at the @ei-th epoch.
    :param signum: the signal number.
    :param ei: the number of the epochs from the beginning of the signal.
    :param freq: the sampling frequency of the signal.
    :param ne: the number of epochs to return.

    :return sig: the datapoints starting at epoch @ei and ending at epoch [@ei + @ne]
    """
    # Number of points in epoch: 30 seconds * frames / second
    npoints = 30 * freq
    # Number of points in @ne epochs
    tot_npoints = npoints * ne
    # Index of point where the epoch starts
    start = ei * npoints
    if mute is False: print "Returning signal in epochs [%d -> %d]" %(ei, ei+ne)

    return sig[ start : start + tot_npoints]

def get_epoch_times(ei, ne, freq):
    """This function returns the starting and ending time indices of the signal
    for the data starting @ei-th epoch and lasting @ne epochs, with @freq frequency.
    :param ei: the number of epochs from the beginning of the signal.
    :param ne: the number of epochs to return.
    :param freq: the frequency of the signal.

    :return: the range, in seconds, corresponding to the timeline of the requested epochs.
    """

    # Number of points in epoch: 30 seconds * frames / second
    npoints = 30 * freq
    # Number of points in @ne epochs
    tot_npoints = npoints * ne
    # Index of point where the epoch starts
    start = ei * npoints
    # Index of point where the epoch ends
    end = int(start + tot_npoints)
    print "Returing timeline [%d : %d]" %(start, end)
    return arange(start, end)

def number_of_epochs(sig, freq):
    """This function returns the number of epochs in the dataset.
    Works with any signal @sig and its corresponding frequency @freq.
    :param sig
    :param freq

    :return: the number of epochs in the dataset
    """

    # Total duration of the signal in seconds
    tot_seconds = len(sig) / freq
    # Number of epochs
    return tot_seconds / 30
