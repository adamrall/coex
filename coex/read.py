"""Functions for reading gchybrid output files."""

import numpy as np


def read_bz(path):
    """Read the activity fractions and beta values of a TEE simulation.

    Args:
        path: The location of the 'bz.dat' file to read.

    Returns:
        A dict containing numpy arrays of beta and the activity
        fractions.
    """
    # Truncate the first column, which just contains an index, read
    # beta separately, and transpose the rest.
    beta = np.loadtxt(path, usecols=(1, ))
    zz = np.transpose(np.loadtxt(path))[2:]

    return {'beta': beta, 'fractions': zz}


def read_lnpi_op(path):
    """Read the order parameter free energy from an lnpi_op.dat file.

    Args:
        path: The location of the file.

    Returns:
        A dict with the order parameter values and free energy.
    """
    index, lnpi = np.loadtxt(path, usecols=(0, 1), unpack=True)

    return {'index': index, 'lnpi': lnpi}


def read_lnpi_tr(path):
    """Read the growth expanded ensemble free energy from an
    lnpi_tr.dat file.

    Args:
        path: The location of the file.

    Returns:
        A dict containing the index, molecule number, stage number,
        and free energy of each entry in the expanded ensemble growth
        path.
    """
    index, sub, mol, stage, lnpi = np.loadtxt(path, usecols=(0, 1, 2, 3, 4),
                                              unpack=True)

    return {'index': index, 'sub': sub, 'mol': mol, 'stage': stage,
            'lnpi': lnpi}


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
