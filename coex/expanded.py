# TODO: Finish porting egda.gcee code.
# TODO: Finish writing documentation.
"""Analyze grand canonical expanded ensemble simulations."""

from __future__ import division
import glob
import os.path

import numpy as np

from coex.read import read_lnpi, read_histogram


def average_histogram(histogram, weights):
    def average_visited_states(states, weight):
        shifted = states['counts'] * np.exp(-weight * states['bins'])

        return sum(shifted * states['bins']) / sum(shifted)

    return np.array([average_visited_states(*pair)
                     for pair in zip(histogram, weights)])


def get_pressure(distribution, volume, beta, histogram=None, is_tee=False):
    lnpi = distribution['logp']
    if histogram is None:
        return lnpi / volume / beta

    size = len(lnpi)
    pressure = np.zeros(size)
    gp = -lnpi / volume / beta
    if is_tee:
        limit = size - 1
        iter_range = reversed(range(size))
    else:
        limit = 0
        iter_range = range(size)

    for i in iter_range:
        dist = histogram[i]
        if dist['bins'][0] < 1.0e-8 and dist['counts'][0] > 1000:
            zero_state = dist['counts'][0] / sum(dist['counts'])
            pressure[i] = -np.log(zero_state) / volume / beta[i]
        else:
            if i == limit:
                pressure[i] = lnpi[i] / beta[i] / volume
            else:
                if is_tee:
                    pressure[i] = pressure[i + 1] + gp[i + 1] - gp[i]
                else:
                    pressure[i] = pressure[i - 1] + gp[i - 1] - gp[i]

    return pressure


def read_all_molecule_histograms(directory):
    hist_files = sorted(glob.glob(os.path.join(directory, "nhist_??.dat")))
    lim_files = sorted(glob.glob(os.path.join(directory, "nlim_??.dat")))

    return [read_histogram(*pair) for pair in zip(hist_files, lim_files)]


def read_bz(path):
    beta = np.loadtxt(path, usecols=(1, ))
    zz = np.transpose(np.loadtxt(path))[2:]

    return {'beta': beta, 'fractions': zz}


def read_zz(path):
    # Truncate the first column, which just contains an index, and
    # transpose the rest.
    return np.transpose(np.loadtxt(path))[1:]


def shift_activity(states, ratio):
    """Find the shift in free energy due to a change in the activity
    of a species.

    Args:
        states: A dict with the keys 'bins' and 'counts': the
            molecule number visited states distribution.
        ratio: The ratio of the new activity to the old activity.

    Returns:
        The shift in the free energy as a float.
    """
    bins, counts = states['bins'], states['counts']

    return (np.log(sum(counts * ratio ** (bins - bins[0]))) -
            np.log(sum(counts)) + bins[0] * np.log(ratio))


def shift_beta(states, difference):
    """Find the shift in free energy due to a change in beta.

    Args:
        states: A dict with the keys 'bins' and 'counts': the energy
            visited states distribution.
        difference: The difference in beta (1 / kT).

    Returns:
        A float corresponding to the shift in the free energy.
    """
    bins, counts = states['bins'], states['counts']
    if np.abs(difference) >= 1e15:
        return (np.log(sum(counts * np.exp(-difference * bins))) -
                np.log(sum(counts)))

    return 0.0
