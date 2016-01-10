import matplotlib.pyplot as plt
import numpy as np
import pandas

#########################
# _Brewer
#########################
class _Brewer(object):
    """Encapsulates a nice sequence of colors.

    Shades of red that look good in color and can be distinguished
    in grayscale (up to a point).

    Borrowed from http://colorbrewer2.org/
    """
    color_iter = None

    colors = ['#081D58',
              '#253494',
              '#225EA8',
              '#1D91C0',
              '#41B6C4',
              '#7FCDBB',
              '#C7E9B4',
              '#EDF8B1',
              '#FFFFD9']

    # lists that indicate which colors to use depending on how many are used
    which_colors = [[],
                    [1],
                    [1, 3],
                    [0, 2, 4],
                    [0, 2, 4, 6],
                    [0, 2, 3, 5, 6],
                    [0, 2, 3, 4, 5, 6],
                    [0, 1, 2, 3, 4, 5, 6],
                    ]

    @classmethod
    def Colors(cls):
        """Returns the list of colors.
        """
        return cls.colors

    @classmethod
    def ColorGenerator(cls, n):
        """Returns an iterator of color strings.

        n: how many colors will be used
        """
        for i in cls.which_colors[n]:
            yield cls.colors[i]
        raise StopIteration('Ran out of colors in _Brewer.ColorGenerator')

    @classmethod
    def InitializeIter(cls, num):
        """Initializes the color iterator with the given number of colors."""
        cls.color_iter = cls.ColorGenerator(num)

    @classmethod
    def ClearIter(cls):
        """Sets the color iterator to None."""
        cls.color_iter = None

    @classmethod
    def GetIter(cls):
        """Gets the color iterator."""
        if cls.color_iter is None:
            cls.InitializeIter(7)

        return cls.color_iter

#########################
# _UnderrideColor
#########################
def _UnderrideColor(options):
    if 'color' in options:
        return options

    color_iter = _Brewer.GetIter()

    if color_iter:
        try:
            options['color'] = next(color_iter)
        except StopIteration:
            # TODO: reconsider whether this should warn
            # warnings.warn('Warning: Brewer ran out of colors.')
            _Brewer.ClearIter()
    return options

#########################
# _Underride
#########################
def _Underride(d, **options):
    """Add key-value pairs to d only if key is not in d.

    If d is None, create a new dictionary.

    d: dictionary
    options: keyword args to add to d
    """
    if d is None:
        d = {}

    for key, val in options.items():
        d.setdefault(key, val)

    return d

#########################
# Plot
#########################
def Plot(obj, ys=None, style='', **options):
    """Plots a line.

    Args:
      obj: sequence of x values, or Series, or anything with Render()
      ys: sequence of y values
      style: style string passed along to plt.plot
      options: keyword args passed to plt.plot
    """
    options = _UnderrideColor(options)
    label = getattr(obj, 'label', '_nolegend_')
    options = _Underride(options, linewidth=3, alpha=0.8, label=label)

    xs = obj
    if ys is None:
        if hasattr(obj, 'Render'):
            xs, ys = obj.Render()
        if isinstance(obj, pandas.Series):
            ys = obj.values
            xs = obj.index

    if ys is None:
        plt.plot(xs, style, **options)
    else:
        plt.plot(xs, ys, style, **options)


# Export function names
plot = Plot
