"""Functions for reading gchybrid output files."""

from __future__ import division

import numpy as np


def read_bz(path):
    """Read the activity fractions and beta values of a TEE simulation.

    Args:
        path: The location of the 'bz.dat' file to read.

    Returns:
        A (beta, activity fractions) tuple of numpy arrays.
    """
    # Truncate the first column, which just contains an index, read
    # beta separately, and transpose the rest.
    beta = np.loadtxt(path, usecols=(1, ))
    zz = np.transpose(np.loadtxt(path))[2:]

    return beta, zz


def read_order_parameter(path):
    """Read the simulation order parameter from an lnpi_op.dat file.

    Args:
        path: The location of the file.

    Returns:
        A numpy array.
    """
    return np.loadtxt(path, usecols=(0, ))


def read_lnpi(path):
    """Read the logarithmic probability distribution from a file.

    Args:
        path: The location of the lnpi_op.dat file.

    Returns:
        A numpy array.
    """
    return np.loadtxt(path, usecols=(1, ))


def read_zz(path):
    """Read the activity fractions of an AFEE simulation.

    Args:
        path: The location of the 'zz.dat' file to read.

    Returns:
        A numpy array: the first row contains the logarithm of the sum
        of the activities for each subensemble, and each subsequent
        row contains the activity fractions of each species after the
        first.
    """
    # Truncate the first column, which just contains an index, and
    # transpose the rest.
    return np.transpose(np.loadtxt(path))[1:]
