"""A class and helper functions for dealing with density
histograms.
"""

import glob
import os.path

import numpy as np


class DensityHistogram(object):
    """Read, combine, and write density histograms (pzhist_*.dat).

    Attributes:
        index: A numpy array enumerating the subensembles of the
            simulation.
        distances: An array of z-coordinate bins for the histogram.
        histogram: A 2D array of the relative densities at each
            distance bin for each subensemble.
        frequencies: An array with the number of samples for each
            subensemble.
    """

    def __init__(self, index, distances, histogram, frequencies):
        self.index = index
        self.distances = distances
        self.histogram = histogram
        self.frequencies = frequencies

    @classmethod
    def from_file(cls, path):
        """Read a density histogram from a pzhist*.dat file.

        Note that the index and frequencies are read from the
        pzcnt.dat file in the same directory as the pzhist file.

        Args:
            path: The location of the pzhist_*.dat file.

        Returns:
            A DensityHistogram object.
        """
        histogram = np.transpose(np.loadtxt(path))
        dirname = os.path.dirname(path)
        index, frequencies = np.loadtxt(os.path.join(dirname, 'pzcnt.dat'),
                                        unpack=True)

        return cls(index=index, distances=histogram[1],
                   histogram=histogram[2:], frequencies=frequencies)

    @classmethod
    def from_combination(cls, hists):
        """Create a density histogram by averaging a list of provided
        histograms.

        Args:
            hists: A list of DensityHistograms.

        Returns:
            A new DensityHistogram with the apropriately summed
            frequencies and averaged densities.
        """
        index, distances = hists[0].index, hists[0].distances
        freq_sum = sum([h.frequencies for h in hists])
        weighted_hist = sum([h.frequencies * np.transpose(h.histogram)
                             for h in hists])
        hist = np.transpose(np.nan_to_num(weighted_hist / freq_sum))

        return DensityHistogram(index=index, distances=distances,
                                histogram=hist, frequencies=freq_sum)

    @classmethod
    def from_combined_runs(cls, path, runs):
        """Combine the density histograms of multiple production
        runs.

        Args:
            path: The directory containing the runs to combine.
            runs: The list of runs to combine.

        Returns:
            A DensityHistogram object with the summed freqeuencies
            and averaged densities of all the runs.
        """
        return cls.from_combination([cls.from_file(os.path.join(path, r))
                                     for r in runs])

    def write(self, path, write_pzcnt=False):
        """Write a density histogram to a pzhist_*.dat file.

        Args:
            path: The name of the file to write.
            write_pzcnt: If True, also write a pzcnt.dat file in the
                same directory with the frequencies.
        """
        dirname = os.path.dirname(path)
        if write_pzcnt:
            np.savetxt(os.path.join(dirname, 'pzcnt.dat'),
                       np.column_stack((self.index, self.frequencies)))

        with open(path, 'w') as f:
            for i, col in enumerate(np.transpose(self.histogram)):
                print(i, self.distances(i), col, file=f)


def read_pzhist(path):
    """Read a density histogram from a pzhist*.dat file.

    Note that the index and frequencies are read from the
    pzcnt.dat file in the same directory as the pzhist file.

    Args:
        path: The location of the pzhist_*.dat file.

    Returns:
        A DensityHistogram object.
    """
    return DensityHistogram.from_file(path)


def read_all_pzhists(path):
    """Read the density histograms for all atoms of all species.

    Args:
        path: The directory containing the pzhist_*.dat files.

    Returns:
        A dict of DensityHistogram objects, with the names of each
        file as the keys.
    """
    files = sorted(glob.glob(os.path.join(path, 'pzhist_*.dat')))

    return {f: read_pzhist(f) for f in files}


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
    files = sorted(glob.glob(os.path.join(path, runs[0], 'pzhist_*.dat')))

    return {f: DensityHistogram.from_combined_runs(f, runs) for f in files}
