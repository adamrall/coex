"""Functions for working with visited states distributions.

Visited states distributions are used to find the change in free
energy via histogram reweighting. For example, an energy visited
states distribution would provide a new free energy given a change
in the thermodynamic beta (1/kT).
"""

import copy
import glob
import os.path

import numpy as np


# The conversion factor for cubic angstroms -> cubic meters.
cubic_meters = 1.0e-30


class VisitedStatesDistribution(object):
    def __init__(self, bins, counts):
        self.bins = bins
        self.counts = counts

    def __len__(self):
        return len(self.bins)

    def __str__(self):
        string = "bins       counts\n"
        if np.issubdtype(self.bins.dtype, np.integer):
            return string + "\n".join(
                ["{:<9d}  {:<}".format(b, self.counts[i])
                 for i, b in enumerate(self.bins)])
        else:
            return string + "\n".join(
                ["{:<.2e}  {:<}".format(b, self.counts[i])
                 for i, b in enumerate(self.bins)])

    def __repr__(self):
        return str(self)

    def reweight(self, amount):
        """Get the change in free energy due to histogram reweighting
        of a visited states distribution.

        Args:
            amount: The difference in the relevant property.

        Returns:
            The change in free energy as a float.
        """
        shifted = self.shift(amount)

        return np.log(sum(shifted)) - np.log(sum(self.counts))

    def shift(self, amount):
        """Transform a visited states distribution by a given amount.

        This function returns the weighted counts of a distribution
        as specified by the formula C*exp(-amount*B), where C and B
        are the counts and bins of the distribution, respectively.

        Args:
            amount: The difference in the relevant property.

        Returns:
            A numpy array with the shifted distribution's counts.
        """
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

        shifted = self.shift(weight)

        return sum(self.bins * shifted) / sum(shifted)


class VisitedStatesHistogram(object):
    def __init__(self, distributions):
        self.distributions = distributions

    def __getitem__(self, index):
        return self.distributions[index]

    def __iter__(self):
        for d in self.distributions:
            yield d

    def __len__(self):
        return len(self.distributions)

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

    def write(self, path):
        most_sampled = 0
        with open(_get_limits_path(path), 'w') as f:
            for i, d in enumerate(self):
                sampled = len(d)
                most_sampled = max(most_sampled, sampled)
                min_bin = np.amin(d.bins)
                max_bin = np.amax(d.bins)
                step = d.bins[1] - d.bins[0]
                print('{:8d} {:7d} {:15.7f} {:15.7f} {:15.7f}'.format(
                    i, sampled, min_bin, max_bin, step), file=f)

        raw_hist = np.zeros([most_sampled, len(self) + 1])
        raw_hist[:, 0] = range(1, most_sampled + 1)
        for i, d in enumerate(self):
            sampled = len(d)
            raw_hist[0:sampled, i + 1] = d.counts

        np.savetxt(path, raw_hist, fmt='%8.d', delimiter='  ')


class VolumeVisitedStatesHistogram(VisitedStatesHistogram):
    def __init__(self, distributions, use_log_volume=False):
        self.use_log_volume = use_log_volume
        super().__init__(distributions)

        for d in self:
            if use_log_volume:
                d.bins = np.exp(d.bins)

            d.bins *= cubic_meters

    def write(self, path):
        copy_with_units = copy.copy(self)
        for d in copy_with_units:
            if d.use_log_volume:
                d.bins = np.log(d.bins)

            d.bins /= cubic_meters

        super(VisitedStatesHistogram, copy_with_units).write(path)


def combine_histograms(hists):
    """Combine a set of visited states histograms.

    Args:
        hists: A list of histograms.

    Returns:
        A histogram with the combined data.
    """
    first_dist = hists[0][0]
    step = first_dist.bins[1] - first_dist.bins[0]
    subensembles = len(first_dist)

    def combine_subensemble(i):
        min_bin = min([h[i].bins[0] for h in hists])
        max_bin = max([h[i].bins[-1] for h in hists])
        num = int((max_bin - min_bin) / step) + 1
        bins = np.linspace(min_bin, max_bin, num, dtype=hists[0][i].bins.dtype)
        counts = np.zeros(num, dtype=hists[0][i].counts.dtype)
        for h in hists:
            shift = int((h[i].bins[0] - min_bin) / step)
            counts[shift:(shift + len(h[i]))] += h[i].counts

        return VisitedStatesDistribution(bins=bins, counts=counts)

    distributions = [combine_subensemble(i) for i in range(subensembles)]
    if isinstance(hists[0], VolumeVisitedStatesHistogram):
        return VolumeVisitedStatesHistogram(distributions,
                                            hists[0].use_log_volume)

    return VisitedStatesHistogram(distributions)


def read_all_molecule_histograms(path):
    """Read all of the molecule number visited states histograms in a
    directory.

    Args:
        path: The directory containing the the nhist and nlim files
            to read.

    Returns:
        A list of histograms as read by read_histogram. Each one is
        itself a list of dicts with the keys 'bins' and 'counts'.

    See Also:
        read_histogram()
    """
    hist_files = sorted(glob.glob(os.path.join(path, "nhist_??.dat")))

    return [read_histogram(hist) for hist in hist_files]


def read_energy_distribution(path, subensemble):
    """Read an energy visited states distribution.

    This function reads the energy visited states histogram from a
    specified directory but returns only the distribution for a
    specified subensemble.  It is used for reweighting a subensemble
    in a TEE simulation to a reference point of different beta.

    Args:
        path: The directory containing the ehist and elim files to
            read.
        subensemble: The specific distribution requested.

    Returns:
        A distribution, i.e., a dict with the keys 'bins' and
        'counts'.
    """
    return read_histogram(os.path.join(path, 'ehist.dat'))[subensemble]


def _get_limits_path(hist_file):
    return os.path.join(os.path.dirname(hist_file),
                        os.path.basename(hist_file).replace('hist', 'lim'))


def read_histogram(path):
    """Read a visited states histogram from a pair of files.

    Args:
        path: The location of the raw histogram data.

    Returns:
        A histogram, i.e., a list of distributions.
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

    return VisitedStatesHistogram([create_distribution(line)
                                   for line in limits])


def read_histograms_from_runs(path, runs, hist_name):
    return [read_histogram(os.path.join(path, r, hist_name))
            for r in sorted(runs)]


def read_volume_histogram(path, use_log_volume=False):
    """Read a volume visited states histogram from a pair of files.

    Args:
        path: The location of the raw histogram data.
        use_log_volume: A bool denoting whether the histogram bins
            store the volume or the logarithm of the volume.

    Returns:
        A histogram, i.e., a list of distributions.
    """
    return VolumeVisitedStatesHistogram(read_histogram(path).distributions,
                                        use_log_volume)


def read_volume_histograms_from_runs(path, runs, use_log_volume=False):
    return [read_volume_histogram(os.path.join(path, r, 'vhist.dat'),
                                  use_log_volume)
            for r in sorted(runs)]
