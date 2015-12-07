# TODO: Finish writing documentation.
"""Functions for reading gchybrid output files."""

from __future__ import division
import glob
import os.path

import numpy as np


def read_all_molecule_histograms(directory):
    """Read all of the molecule number visited states histograms in a
    directory.

    Args:
        directory: The location of the nhist and nlim files to read.

    Returns:
        A list of histograms as read by read_histogram. Each one is
        itself a list of dicts with the keys 'bins' and 'counts'.

    See Also:
        read_histogram()
    """
    hist_files = sorted(glob.glob(os.path.join(directory, "nhist_??.dat")))
    lim_files = sorted(glob.glob(os.path.join(directory, "nlim_??.dat")))

    return [read_histogram(*pair) for pair in zip(hist_files, lim_files)]


def read_bz(path):
    # Truncate the first column, which just contains an index, read
    # beta separately, and transpose the rest.
    beta = np.loadtxt(path, usecols=(1, ))
    zz = np.transpose(np.loadtxt(path))[2:]

    return {'beta': beta, 'fractions': zz}


def read_energy_distribution(directory, subensemble):
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
        return {'bins': bins, 'counts': counts}

    return [parse_limits(line) for line in limits]


def read_lnpi(path):
    """Read the logarithmic probability distribution from a file.

    Args:
        path: The location of the lnpi_op.dat file.

    Returns:
        A dict with the keys 'param' and 'logp', each referring to a
        numpy array.
    """
    param, logp = np.loadtxt(path, usecols=(0, 1), unpack=True)

    return {'param': param, 'logp': logp}


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
            dist['bins'] = np.exp(dist['bins'])

        dist['bins'] *= cubic_meters

    return hist


def read_zz(path):
    # Truncate the first column, which just contains an index, and
    # transpose the rest.
    return np.transpose(np.loadtxt(path))[1:]
