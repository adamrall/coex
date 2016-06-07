"""Objects and functions for working with visited states
distributions.

Visited states distributions are used to find the change in free
energy via histogram reweighting. For example, an energy visited
states distribution would provide a new free energy given a change in
the thermodynamic beta (1/kT).
"""

import copy
import glob
import os.path

import numpy as np


# The conversion factor for cubic angstroms -> cubic meters.
cubic_meters = 1.0e-30


class VisitedStatesDistribution(object):
    """A frequency distribution for a given property (energy, volume,
    molecule count) of a single subensemble of a simulation.

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
        """Calculate the weighted average of a visited states
        distribution.

        Args:
            weight: The optional weight to use.

        Returns:
            The weighted average as a float.
        """
        if weight is None:
            return sum(self.counts * self.bins) / sum(self.counts)

        shifted = self._shift(weight)

        return sum(self.bins * shifted) / sum(shifted)

    @staticmethod
    def from_file(path, subensemble):
        """Read a single distribution from a pair of *hist*.dat and
        *lim*.dat files.

        Args:
            path: The location of the *hist*.dat file.
            subensemble: The subensemble number of the distribution.

        Returns:
            A VisitedStatesDistribution object.
        """
        if 'vhist' in os.path.basename(path):
            return VolumeVisitedStatesHistogram.from_file(path)[subensemble]

        return VisitedStatesHistogram.from_file(path)[subensemble]


def _get_limits_path(hist_file):
    return os.path.join(os.path.dirname(hist_file),
                        os.path.basename(hist_file).replace('hist', 'lim'))


class VisitedStatesHistogram(object):
    """A list of visited states distributions, each corresponding to one
    subensemble of the simulation.

    Attributes:
        dists: A list of VisitedStatesDistribution objects.
    """

    def __init__(self, dists):
        self.dists = dists

    def __getitem__(self, index):
        return self.dists[index]

    def __iter__(self):
        for d in self.dists:
            yield d

    def __len__(self):
        return len(self.dists)

    def average(self, weights=None):
        """Calculate the weighted average of the histogram.

        Args:
            weights: A list of weights for each distribution in the
                histogram.

        Returns:
            A numpy array with the weighted average of each distribution
            in the histogram.
        """
        if weights is None:
            return np.array([d.average(weight=None) for d in self])

        return np.array([d.average(weights[i]) for i, d in self])

    @classmethod
    def from_combination(cls, hists):
        """Combine a set of visited states histograms.

        Args:
            hists: A list of histograms.

        Returns:
            A VisitedStatesHistogram with the combined data.
        """
        first_dist = hists[0][0]
        step = first_dist.bins[1] - first_dist.bins[0]
        subensembles = len(first_dist)

        def combine_subensemble(i):
            min_bin = min([h[i].bins[0] for h in hists])
            max_bin = max([h[i].bins[-1] for h in hists])
            num = int((max_bin - min_bin) / step) + 1
            bins = np.linspace(min_bin, max_bin, num,
                               dtype=hists[0][i].bins.dtype)
            counts = np.zeros(num, dtype=hists[0][i].counts.dtype)
            for h in hists:
                shift = int((h[i].bins[0] - min_bin) / step)
                counts[shift:(shift + len(h[i]))] += h[i].counts

            return VisitedStatesDistribution(bins=bins, counts=counts)

        return cls([combine_subensemble(i) for i in range(subensembles)])

    @classmethod
    def from_combined_runs(cls, path, runs, hist_file):
        """Combine a visited states histogram across a series of
        production runs.

        Args:
            path: The location of the production runs.
            runs: The list of runs to combine.
            hist_file: The name of the histogram to combine.

        Returns:
            A VisitedStatesHistogram with the combined data.
        """
        return cls.from_combination(
            [cls.from_file(os.path.join(path, r, hist_file)) for r in runs])

    @classmethod
    def from_file(cls, path):
        """Read a visited states histogram from a pair of files.

        This method accepts the location of the raw histogram file, e.g.,
        ehist.dat and parses the appropriate limits file (here, elim.dat)
        in the same directory.

        Args:
            path: The location of the raw histogram data.

        Returns:
            A VisitedStatesHistogram object.
        """
        raw_hist = np.transpose(np.loadtxt(path))[1:]
        limits = np.loadtxt(_get_limits_path(path))

        def create_distribution(line):
            sub, size, lower, upper, step = line
            sub, size = int(sub), int(size)
            bins = np.linspace(lower, upper, size)
            if 'nhist' in path:
                bins = bins.astype('int')

            if len(raw_hist.shape) == 1:
                counts = np.array([raw_hist[sub]])
            else:
                counts = raw_hist[sub][0:size]

            return VisitedStatesDistribution(bins=bins,
                                             counts=counts.astype('int'))

        return cls([create_distribution(line) for line in limits])

    def write(self, path):
        """Write a histogram to a pair of hist and lim files.

        Args:
            path: The name of the *hist*.dat file to write.
        """
        most_sampled = 0
        step = self[-1].bins[1] - self[-1].bins[0]
        with open(_get_limits_path(path), 'w') as f:
            for i, d in enumerate(self):
                sampled = len(d)
                most_sampled = max(most_sampled, sampled)
                min_bin = np.amin(d.bins)
                max_bin = np.amax(d.bins)
                print('{:8d} {:7d} {:15.7e} {:15.7e} {:15.7e}'.format(
                    i, sampled, min_bin, max_bin, step), file=f)

        raw_hist = np.zeros([most_sampled, len(self) + 1])
        raw_hist[:, 0] = range(1, most_sampled + 1)
        for i, d in enumerate(self):
            sampled = len(d)
            raw_hist[0:sampled, i + 1] = d.counts

        np.savetxt(path, raw_hist, fmt='%8.d', delimiter='  ')


def read_ehist(path):
    """Read an energy histogram from an ehist.dat file."""
    return VisitedStatesHistogram.from_file(path)


def read_nhist(path):
    """Read a molecule number histogram from an nhist_*.dat file."""
    return VisitedStatesHistogram.from_file(path)


def read_all_nhists(path):
    """Read all of the molecule number visited states histograms in a
    directory.

    Args:
        path: The directory containing the the nhist and nlim files
            to read.

    Returns:
        A sorted list of VisitedStatesHistogram objects.
    """
    hist_files = sorted(glob.glob(os.path.join(path, "nhist_??.dat")))

    return [VisitedStatesHistogram.from_file(f) for f in hist_files]


def combine_all_nhists(path, runs):
    """Combine all molecule number visited states histograms from a
    set of runs.

    Args:
       path: The base path containing the data to combine.
       runs: The list of runs to combine.

    Returns:
        A list of combined histograms, with one entry for each
        species.
    """
    hist_files = [os.path.basename(f)
                  for f in glob.glob(os.path.join(runs[0], 'nhist_*.dat'))]

    return [VisitedStatesHistogram.from_combined_runs(path, runs, hf)
            for hf in hist_files]


class VolumeVisitedStatesHistogram(VisitedStatesHistogram):
    """A list of volume visited states distributions.

    Attributes:
        dists: The list of VisitedStatesDistribution objects.
    """

    def __init__(self, dists):
        super(VolumeVisitedStatesHistogram, self).__init__(dists)

    def _adjust_units(self, use_log_volume=False):
        for d in self:
            if use_log_volume:
                d.bins = np.exp(d.bins)

            d.bins *= cubic_meters

    @classmethod
    def from_combined_runs(cls, path, runs, use_log_volume=False):
        """Construct a volume histogram by combining the histograms
        from several production runs.

        Args:
            path: The location of the production runs.
            runs: The list of runs to combine.
            use_log_volume: A bool; True if the vhist.dat file has
                ln(V) bins instead of volume bins.

        Returns:
            A VolumeVisitedStatesHistogram with the combined data.
        """
        hist = VisitedStatesHistogram.from_combined_runs(path, runs,
                                                         'vhist.dat')

        return cls(hist.dists)._adjust_units(use_log_volume)

    @classmethod
    def from_file(cls, path, use_log_volume=False):
        """Read a volume histogram from a vhist.dat file.

        Args:
            path: The location of the file.
            use_log_volume: A bool; True if the file has ln(V) bins
                instead of volume bins.

        Returns:
            A VolumeVisitedStatesHistogram object.
        """
        hist = VisitedStatesHistogram.from_file(path)

        return cls(hist.dists)._adjust_units(use_log_volume)

    def write(self, path, use_log_volume=False):
        """Write the histogram to a pair of vhist.dat and vlim.dat
        files.

        Args:
            path: The location of the hist file.
            use_log_volume: A bool; True if the lim file should be
                written using ln(V) bins instead of volume bins.
        """
        units_copy = copy.copy(self)
        for d in units_copy:
            if units_copy.use_log_volume:
                d.bins = np.log(d.bins)

            d.bins /= cubic_meters

        super(VisitedStatesHistogram, units_copy).write(path)


def read_vhist(path, use_log_volume):
    """Read a volume visited states histogram from a vhist.dat
    file.
    """
    return VolumeVisitedStatesHistogram.from_file(path, use_log_volume)
