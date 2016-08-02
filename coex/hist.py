"""Objects and functions for working with visited states histograms.

Histograms are used to find the change in free energy via histogram
reweighting.  For example, an energy visited states distribution would
provide a new free energy given a change in the inverse temperature
beta (1/kT).
"""

import glob
import os.path

import numpy as np


# The conversion factor for cubic angstroms -> cubic meters.
CUBIC_METERS = 1.0e-30


class SubHistogram(object):
    """A frequency distribution for a given property (energy, volume,
    molecule count) in a single subensemble of a simulation.

    Attributes:
        bins: A numpy array with the values of the property.
        counts: An array with the number of times each value was
            visited in the simulation.
    """

    def __init__(self, bins, counts):
        self.bins = bins
        self.counts = counts

    def __len__(self):
        return len(self.bins)

    def __str__(self):
        return "bins: {}, counts: {}\n".format(self.bins, self.counts)

    def __repr__(self):
        return str(self)

    def reweight(self, amount):
        """Get the change in free energy due to histogram reweighting.

        Args:
            amount: The difference in the relevant property.

        Returns:
            The change in free energy as a float.
        """
        shifted = self._shift(amount)

        return np.log(sum(shifted)) - np.log(sum(self.counts))

    def _shift(self, amount):
        return self.counts * np.exp(-amount * self.bins)

    def average(self, weight=None):
        """Calculate the weighted average of the histogram.

        Args:
            weight: The optional weight to use.

        Returns:
            The weighted average as a float.
        """
        if weight is None:
            return sum(self.counts * self.bins) / sum(self.counts)

        shifted = self._shift(weight)

        return sum(self.bins * shifted) / sum(shifted)


class Histogram(object):
    """A list of subensemble-specific visited states histograms for a
    given property.

    Attributes:
        subhists: A list of SubHistogram objects.
    """

    def __init__(self, subhists):
        self.subhists = subhists

    def __getitem__(self, index):
        return self.subhists[index]

    def __iter__(self):
        for d in self.subhists:
            yield d

    def __len__(self):
        return len(self.subhists)

    def average(self, weights=None):
        """Calculate the weighted average of the histogram.

        Args:
            weights: A list of weights for each subensemble.

        Returns:
            A numpy array with the weighted average of each
            subensemble-specific histogram.
        """
        if weights is None:
            return np.array([d.average(weight=None) for d in self])

        return np.array([d.average(weights[i]) for i, d in enumerate(self)])

    def write(self, path):
        """Write a histogram to a pair of hist and lim files.

        Args:
            path: The name of the *hist*.dat file to write.
        """
        hist_file = os.path.basename(path)
        most_sampled = 0
        step = self[-1].bins[1] - self[-1].bins[0]
        if 'vhist' in hist_file:
            step /= CUBIC_METERS

        with open(_get_limits_path(path), 'w') as f:
            for i, sh in enumerate(self):
                sampled = len(sh)
                most_sampled = max(most_sampled, sampled)
                min_bin = np.amin(sh.bins)
                max_bin = np.amax(sh.bins)
                if 'vhist' in hist_file:
                    max_bin /= CUBIC_METERS
                    min_bin /= CUBIC_METERS

                print('{:8d} {:7d} {:15.7e} {:15.7e} {:15.7e}'.format(
                    i, sampled, min_bin, max_bin, step), file=f)

        raw_hist = np.zeros([most_sampled, len(self) + 1])
        raw_hist[:, 0] = range(1, most_sampled + 1)
        for i, sh in enumerate(sh):
            sampled = len(sh)
            raw_hist[0:sampled, i + 1] = sh.counts

        np.savetxt(path, raw_hist, fmt='%8d', delimiter='  ')


def _get_limits_path(hist_file):
    return os.path.join(os.path.dirname(hist_file),
                        os.path.basename(hist_file).replace('hist', 'lim'))


def read_hist(path):
    """Read a histogram from a pair of files.

    This method accepts the location of the raw histogram file, e.g.,
    ehist.dat and parses the appropriate limits file (here, elim.dat)
    in the same directory.

    Args:
        path: The location of the raw histogram data.

    Returns:
        A Histogram object.
    """
    raw_hist = np.transpose(np.loadtxt(path))[1:]
    limits = np.loadtxt(_get_limits_path(path))

    def create_subhist(line):
        sub, size, lower, upper, step = line
        sub, size = int(sub), int(size)
        bins = np.linspace(lower, upper, size)
        if 'nhist' in path:
            bins = bins.astype('int')

        if len(raw_hist.shape) == 1:
            counts = np.array([raw_hist[sub]])
        else:
            counts = raw_hist[sub][0:size]

        return SubHistogram(bins=bins, counts=counts.astype('int'))

    return Histogram([create_subhist(line) for line in limits])


def _normalize_vhist_units(hist, use_log_volume):
    for sh in hist:
        if use_log_volume:
            sh.bins = np.exp(sh.bins)

        sh.bins *= CUBIC_METERS


def read_vhist(path, use_log_volume=False):
    """Read a volume histogram from a vhist.dat file.

    Args:
        path: The location of the histogram data.
        use_log_volume: A bool; True if the lim file uses ln(V) bins
            instead of volume bins.
    """
    return _normalize_vhist_units(read_hist(path), use_log_volume)


def combine_hists(hists):
    """Combine a series of histograms.

    Args:
        hists: A list of histograms.

    Returns:
        A Histogram with the combined data.
    """
    fst_sub = hists[0][0]
    step = fst_sub.bins[1] - fst_sub.bins[0]
    subensembles = len(fst_sub)
    dtype = fst_sub.bins.dtype

    def combine_subensemble(i):
        min_bin = min([h[i].bins[0] for h in hists])
        max_bin = max([h[i].bins[-1] for h in hists])
        num = int((max_bin - min_bin) / step) + 1
        bins = np.linspace(min_bin, max_bin, num, dtype=dtype)
        counts = np.zeros(num, dtype=dtype)
        for h in hists:
            shift = int((h[i].bins[0] - min_bin) / step)
            counts[shift:(shift + len(h[i]))] += h[i].counts

        return SubHistogram(bins=bins, counts=counts)

    return Histogram([combine_subensemble(i) for i in range(subensembles)])


def combine_hist_runs(path, runs, hist_file):
    """Combine histograms across a series of production runs.

    Args:
        path: The location of the production runs.
        runs: The list of runs to combine.
        hist_file: The name of the histogram to combine.

    Returns:
        A Histogram with the combined data.
    """
    return combine_hists([read_hist(os.path.join(path, r, hist_file))
                          for r in runs])


def combine_vhist_runs(path, runs, use_log_volume=False):
    """Combine volume histograms across a series of production runs.

    Args:
        path: The location of the production runs.
        runs: The list of runs to combine.
        use_log_volume: A bool; True if the lim file uses ln(V) bins
            instead of volume bins.

    Returns:
        A Histogram with the combined data.
    """
    return _normalize_vhist_units(
        combine_hist_runs(path, runs, hist_file='vhist.dat'), use_log_volume)


def read_all_nhists(path):
    """Read all of the molecule number histograms in a directory.

    Args:
        path: The directory containing the the nhist and nlim files to
            read.

    Returns:
        A sorted list of Histogram objects.
    """
    hist_files = sorted(glob.glob(os.path.join(path, "nhist_??.dat")))

    return [read_hist(f) for f in hist_files]


def combine_all_nhists(path, runs):
    """Combine all the molecule number histograms across a set of runs.

    Args:
       path: The base path containing the data to combine.
       runs: The list of runs to combine.

    Returns:
        A list of combined histograms, with one entry for each species.
    """
    hist_files = (os.path.basename(f)
                  for f in glob.glob(os.path.join(runs[0], 'nhist_*.dat')))

    return [combine_hist_runs(path, runs, hf) for hf in hist_files]
