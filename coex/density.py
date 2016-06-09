"""A class and helper functions for dealing with density
histograms.
"""

import glob
import os.path

import numpy as np


class DensityHistogram(object):
    """A collection of data about the z-direction local density.

    Attributes:
        index: A numpy array enumerating the subensembles of the
            simulation.
        zbins: An array of z-coordinate bins for the histogram.
        rho: A 2D array of the relative densities at each distance bin
            for each subensemble.
        freqs: An array with the number of samples for each
            subensemble.
    """

    def __init__(self, index, zbins, rho, freqs):
        self.index = index
        self.zbins = zbins
        self.rho = rho
        self.freqs = freqs

    def write_pzhist(self, path, write_pzcnt=False):
        """Write a density histogram to a pzhist_*.dat file.

        Args:
            path: The name of the file to write.
            write_pzcnt: If True, also write a pzcnt.dat file in the
                same directory with the frequencies.
        """
        dirname = os.path.dirname(path)
        if write_pzcnt:
            np.savetxt(os.path.join(dirname, 'pzcnt.dat'),
                       np.column_stack((self.index, self.freqs)))

        np.savetxt(path, np.column_stack((self.index,
                                          *np.transpose(self.rho))))


def read_pzhist(path):
    """Read a density histogram from a pzhist*.dat file.

    Note that the index and frequencies are read from the
    pzcnt.dat file in the same directory as the pzhist file.

    Args:
        path: The location of the pzhist_*.dat file.

    Returns:
        A DensityHistogram object.
    """
    hist = np.transpose(np.loadtxt(path))
    dirname = os.path.dirname(path)
    index, freqs = np.loadtxt(os.path.join(dirname, 'pzcnt.dat'), unpack=True)

    return DensityHistogram(index=index, zbins=hist[1], rho=hist[2:],
                            freqs=freqs)


def combine_histograms(hists):
    """Create a density histogram by averaging a list of provided
    histograms.

    Args:
        hists: A list of DensityHistograms.

    Returns:
        A new DensityHistogram with the apropriately summed
        frequencies and averaged densities.
    """
    index, zbins = hists[0].index, hists[0].zbins
    freq_sum = sum([h.freqs for h in hists])
    weighted = sum(h.frequencies * np.transpose(h.rho) for h in hists)
    rho = np.transpose(np.nan_to_num(weighted / freq_sum))

    return DensityHistogram(index=index, zbins=zbins, rho=rho, freqs=freq_sum)


def combine_pzhist_runs(path, runs, hist_file):
    """Combine the density histograms of multiple production
    runs.

    Args:
        path: The directory containing the runs to combine.
        runs: The list of runs to combine.
        hist_file: The specific histogram file to combine.

    Returns:
        A DensityHistogram object with the summed freqeuencies
        and averaged densities of all the runs.
    """
    return combine_histograms([read_pzhist(os.path.join(path, r, hist_file))
                               for r in runs])


def read_all_pzhists(path):
    """Read the density histograms for all atoms of all species.

    Args:
        path: The directory containing the pzhist_*.dat files.

    Returns:
        A dict of DensityHistogram objects, with the names of each
        file as the keys.
    """
    files = sorted(glob.glob(os.path.join(path, 'pzhist_*.dat')))

    return {os.path.basename(f): read_pzhist(f) for f in files}


def combine_all_pzhists(path, runs):
    """Combine the density histograms for all atoms of all species
    from a series of production runs.

    Args:
        path: The directory containing the production runs.
        runs: The list of runs to combine.

    Returns:
        A dict of DensityHistogram objects, with the names of each
        file as the keys.
    """
    files = [os.path.basename(pf)
             for pf in sorted(glob.glob(os.path.join(path, runs[0],
                                                     'pzhist_*.dat')))]

    return {f: combine_pzhist_runs(path, runs, f) for f in files}
