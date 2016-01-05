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

"""Functions for working with visited states distributions.

Each distribution is represented by a dict containing the keys 'bins'
and 'counts'. A collection of distributions is referred to as a
histogram.

Visited states distributions are used to find the change in free
energy via histogram reweighting. For example, an energy visited
states distribution would provide a new free energy given a change
in the thermodynamic beta (1/kT).
"""

from __future__ import division
import glob
import os.path

import numpy as np


def reweight_distribution(dist, amount):
    """Get the change in free energy due to histogram reweighting of
    a visited states distribution.

    Args:
        dist: A distribution.
        amount: The difference in the relevant property.

    Returns:
        The change in free energy as a float.
    """
    shifted = shift_distribution(dist, amount)

    return np.log(sum(shifted)) - np.log(sum(dist['counts']))


def shift_distribution(dist, amount):
    """Transform a visited states distribution by a given amount.

    This function returns the weighted counts of a distribution as
    specified by the formula C*exp(-amount*B), where C and B are the
    counts and bins of the distribution, respectively.

    Args:
        dist: A distribution.
        amount: The difference in the relevant property.

    Returns:
        A numpy array with the shifted distribution's counts.
    """
    return dist['counts'] * np.exp(-amount * dist['bins'])


def average_distribution(dist, weight=None):
    """Calculate the weighted average of a visited states
    distribution.

    Args:
        dist: A distribution.
        weight: The optional weight to use.

    Returns:
        The weighted average as a float.
    """
    if weight is None:
        return sum(dist['counts'] * dist['bins']) / sum(dist['counts'])

    shifted = shift_distribution(dist, weight)

    return sum(dist['bins'] * shifted) / sum(shifted)


def average_histogram(hist, weights=None):
    """Calculate the weighted average of a visited states histogram.

    Args:
        hist: A histogram, i.e., a list of distributions.
        weights: A list of weights for each distribution in the
            histogram.

    Returns:
        A numpy array with the weighted average of each distribution
        in the histogram.
    """
    if weights is None:
        return np.array([average_distribution(dist) for dist in hist])

    return np.array([average_distribution(dist, weights[i])
                     for i, dist in enumerate(hist)])


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
    lim_files = sorted(glob.glob(os.path.join(path, "nlim_??.dat")))

    return [read_histogram(*pair) for pair in zip(hist_files, lim_files)]


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
    hist_file = os.path.join(path, 'ehist.dat')
    lim_file = os.path.join(path, 'elim.dat')

    return read_histogram(hist_file, lim_file)[subensemble]


def read_histogram(hist_path, lim_path):
    """Read a visited states histogram from a pair of files.

    Args:
        hist_path: The location of the raw histogram data.
        lim_path: The location of the limits for the histogram.

    Returns:
        A histogram, i.e., a list of distributions.
    """
    raw_hist = np.transpose(np.loadtxt(hist_path))
    limits = np.loadtxt(lim_path)

    def parse_limits(line):
        sub, size, lower, upper, step = line
        bins = np.linspace(lower, upper, step)
        counts = raw_hist[sub][0:size]

        return {'bins': bins, 'counts': counts}

    return [parse_limits(line) for line in limits]


def read_volume_histogram(hist_path, lim_path, uses_log_volume=False):
    """Read a volume visited states histogram from a pair of files.

    Args:
        hist_path: The location of the raw histogram data.
        lim_path: The location of the limits for the histogram.
        uses_log_volume: A bool denoting whether the histogram bins
            store the volume or the logarithm of the volume.

    Returns:
        A histogram, i.e., a list of distributions.
    """
    # The conversion factor for cubic angstroms -> cubic meters.
    cubic_meters = 1.0e-30

    hist = read_histogram(hist_path, lim_path)
    for dist in hist:
        if uses_log_volume:
            dist['bins'] = np.exp(dist['bins'])

        dist['bins'] *= cubic_meters

    return hist
