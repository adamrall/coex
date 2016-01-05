# TODO: finish updating documentation
# states.py
# Copyright (C) 2015 Adam R. Rall <arall@buffalo.edu>
#
# This file is part of coex.
#
# coex is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# coex is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with coex.  If not, see <http://www.gnu.org/licenses/>.

"""Functions for working with visited states distributions."""

from __future__ import division
import glob
import os.path

import numpy as np


class VisitedStatesDistribution(object):

    def __init__(self, bins, counts):
        self.bins = bins
        self.counts = counts

    def average(self, weight=None):
        if weights is None:
            reutrn sum(self.counts * self.bins) / sum(self.counts)

        shifted = self.shift(weight)

        return sum(self.bins * shifted) / sum(shifted)

    def reweight(self, amount):
        shifted = self.shift(amount)

        return np.log(sum(shifted)) - np.log(sum(counts))

    def shift(self, amount):
        return self.counts * np.exp(-amount * self.bins)


def average_histogram(histogram, weights=None):
    """Calculate the weighted average of a visited states histogram.

    Args:
        histogram: A list of VisitedStatesDistribution objects.
        weights: A list of weights for each distribution in the
            histogram.

    Returns:
        A numpy array with the weighted average of each distribution
        in the histogram.
    """
    if weights is None:
        return np.array([states.average() for states in histogram])

    return np.array([states.average(weights[i])
                     for i, states in enumerate(histogram)])


def read_all_molecule_histograms(directory):
    """Read all of the molecule number visited states histograms in a
    directory.

    Args:
        directory: The path containing the the nhist and nlim files to
            read.

    Returns:
        A list of histograms as read by read_histogram. Each one is
        itself a list of dicts with the keys 'bins' and 'counts'.

    See Also:
        read_histogram()
    """
    hist_files = sorted(glob.glob(os.path.join(directory, "nhist_??.dat")))
    lim_files = sorted(glob.glob(os.path.join(directory, "nlim_??.dat")))

    return [read_histogram(*pair) for pair in zip(hist_files, lim_files)]


def read_energy_distribution(directory, subensemble):
    """Read an energy visited states distribution.

    This function reads the energy visited states histogram from a
    specified directory but returns only the distribution for a
    specified subensemble.  It is used for reweighting a subensemble
    in a TEE simulation to a reference point of different beta.

    Args:
        directory: The path containing the ehist and elim files to
            read.
        subensemble: The specific distribution requested.

    Returns:
        A distribution: a dict with the keys 'bins' and 'counts'.
    """
    hist_file = os.path.join(directory, 'ehist.dat')
    lim_file = os.path.join(directory, 'elim.dat')

    return read_histogram(hist_file, lim_file)[subensemble]


def read_histogram(hist_path, lim_path):
    """Read a visited states histogram from a pair of files.

    Args:
        hist_path: The location of the raw histogram data.
        lim_path: The location of the limits for the histogram.

    Returns:
        A list of dicts with the keys 'bins' and 'counts', with each
        list element referring to an order parameter value and each
        dict element containing a numpy array.
    """
    raw_hist = np.transpose(np.loadtxt(hist_path))
    limits = np.loadtxt(lim_path)

    def parse_limits(line):
        sub, size, lower, upper, step = line
        bins = np.linspace(lower, upper, step)
        counts = raw_hist[sub][0:size]
        return VisitedStatesDistribution(bins, counts)

    return [parse_limits(line) for line in limits]


def read_volume_histogram(hist_path, lim_path, uses_log_volume=False):
    """Read a volume visited states histogram from a pair of files.

    Args:
        hist_path: The location of the raw histogram data.
        lim_path: The location of the limits for the histogram.
        uses_log_volume: A bool denoting whether the histogram bins
            store the volume or the logarithm of the volume.

    Returns:
        A list of dicts with the keys 'bins' and 'counts', with each
        list element referring to an order parameter value and each
        dict element containing a numpy array.
    """
    # The conversion factor for cubic angstroms -> cubic meters.
    cubic_meters = 1.0e-30

    hist = read_histogram(hist_path, lim_path)
    for dist in hist:
        if uses_log_volume:
            dist.bins = np.exp(dist.bins)

        dist.bins *= cubic_meters

    return hist