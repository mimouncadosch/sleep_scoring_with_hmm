import numpy as np
from scipy import stats
from biosignal import Signal

"""
This file contains the functions that extract features from data.
"""

# Source: http://stats.stackexchange.com/questions/50807/features-for-time-series-classification
def get_features(sig, freq):
    """This function computes statistics for the given signal @sig.
    For information on statistical moments: https://en.wikipedia.org/wiki/Moment_(mathematics)
    :param sig: an array containing points that pertain to the signal
    :param freq: the frequency of the signal in Hertz
    """
    s = Signal(sig, freq)
    # Frequency domain (fd)
    fd = s.to_freq_domain()
    max_freq_idx = np.argmax(fd.amps)
    # Frequency with highest amplitude
    max_freq = fd.fs[max_freq_idx]
    min_freq_idx = np.argmin(fd.amps)
    # Frequency with lowest amplitude
    min_freq = fd.fs[min_freq_idx]

    # print "fd.fs", fd.fs
    # print "freqs: min: %f, max: %f"%(min_freq, max_freq)

    # Normalize frequency domain histogram
    nbins = 10
    normalized_fd = normalize_frequencies(fd, nbins)

    # import pdb; pdb.set_trace()
    feat_array = np.array([ np.mean(sig),
                      stats.moment(sig, 2, axis=0), stats.moment(sig, 3, axis=0), stats.moment(sig, 4, axis=0),
                      np.max(sig), np.min(sig),
                      max_freq, min_freq
                      ])

    # return feat_array
    return normalized_fd
    # return np.append(feat_array, normalized_fd)


def normalize_frequencies(spectrum, nfreqs):
    """This function takes a signal (freqs, amps) in the frequency domain and reduces the number of frequencies.
     To do so, this function adds all the amplitudes that fall within frequency interval in the standardized spectrum.
    The number of frequency amplitudes is standardized to @nfreqs for all signals, so that the feature vectors all have the same length.
    :param spectrum: the Spectrum object
    :param nfreqs: the number of frequencies, so the resolution of the spectrum.
    :return: an array of length @nfreqs with the amplitudes of the frequencies in spectrum.fs
    """

    # Array with standard number @nfreqs of frequency amplitudes.
    # The array std_amps has the same minimum and maximum frequencies as spectrum.fs, but only @nfreqs frequencies
    std_amps = []
    div_factor = len(spectrum.fs) / float(nfreqs)
    div_factor = int( np.ceil( div_factor ))

    for i in range(0, len(spectrum.fs), div_factor):
        ival_start = i
        ival_end = i+div_factor-1

        if ival_end > len(spectrum.fs):
            ival_end = len(spectrum.fs)

        # print ival_start, ival_end
        sum_amps = np.sum(spectrum.amps[ival_start: ival_end+1])
        std_amps.append(sum_amps)

    return std_amps










def autocorr(x):
    """This function computes autocorrelation
    http://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation
    """
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]
