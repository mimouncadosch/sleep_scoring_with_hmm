import numpy as np
import copy
import sig_plot

class Signal(object):
    """Signal data in time domain
    :param framerate in frames per second
    """
    def __init__(self, ys, framerate, start=0):
        super(Signal, self).__init__()
        self.ys = ys
        self.framerate = framerate
        self.start = start

    def __len__(self):
        return len(self.ys)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del(self.ys)

    @property
    def ts(self):
        """Times
        :returns NumPy array of times
        """
        n = len(self.ys)
        return np.linspace(0, self.duration, n)

    @property
    def duration(self):
        """Duration of signal
        :returns: float duration in seconds
        """
        return len(self.ys / float(self.framerate))

    def scale(self, factor):
        """Scales the signal by a factor
        :param factor: scale factor
        """
        self.ys *= factor

    def to_freq_domain(self):
        """Computes the frequency domain spectrum of the time signal using FFT.

        : returns instance of Spectrum class
        """
        return Spectrum(self.ys, self.framerate)

    def copy(self):
        """Makes a copy.

        Returns: new Wave
        """
        return copy.deepcopy(self)


class Spectrum(object):
    """
    Creates a Spectrum from a time signal
    """

    def __init__(self, ys, framerate):
        super(Spectrum, self).__init__()

        # hs: NumPy array of complex values
        self.hs = np.fft.rfft(ys)
        self.framerate = framerate
        self.max_freq = framerate / 2.0

        # the frequency for each component of the spectrum depends
        # on whether the length of the wave is even or odd.
        # see http://docs.scipy.org/doc/numpy/reference/generated/
        # numpy.fft.rfft.html
        n = len(self.hs)
        if n%2 == 0:
            max_freq = self.max_freq
        else:
            max_freq = self.max_freq * (n-1) / n

        # Frequency components
        self.fs = np.linspace(0, max_freq, n)

    @property
    def real(self):
        """Returns the real part of the hs (read-only property)."""
        return np.real(self.hs)


    @property
    def amps(self):
        """Returns a sequence of amplitudes (read-only property)."""
        return np.absolute(self.hs)

    def to_time_domain(self):
        """Transforms the spectrum from frequency domain to time domain.
        returns: instance of Signal class
        """
        ys = np.fft.irfft(self.hs)
        return Signal(ys, self.framerate)

    def low_pass(self, cutoff, factor=0):
        """Attenuate frequencies above the cutoff.

        cutoff: frequency in Hz
        factor: what to multiply the magnitude by
        """
        for i in range(len(self.hs)):
            if self.fs[i] > cutoff:
                self.hs[i] *= factor

    def high_pass(self, cutoff, factor=0):
        """Attenuate frequencies below the cutoff.

        cutoff: frequency in Hz
        factor: what to multiply the magnitude by
        """
        for i in range(len(self.hs)):
            if self.fs[i] < cutoff:
                self.hs[i] *= factor


    def plot(self, low=0, high=None, **options):
        """
        Plots amplitude vs frequency.

        low: int index to start at
        high: int index to end at
        """
        sig_plot.plot(self.fs[low:high], self.amps[low:high], **options)

    def copy(self):
        """Makes a copy.

        Returns: new Wave
        """
        return copy.deepcopy(self)