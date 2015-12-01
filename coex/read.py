"""Functions for reading gchybrid output files."""

from __future__ import division
import numpy as np


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


def read_histogram(hist_path, lim_path):
    """Read a visited-states histogram from a pair of files.

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


def read_volume_histogram(hist_path, lim_path, uses_log_volume=False):
    """Read a volume visited-states histogram from a pair of files.

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
