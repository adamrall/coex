# TODO: Finish porting egda.gcee code.
# TODO: Finish writing documentation.
"""Analyze grand canonical expanded ensemble simulations."""

from __future__ import division
import glob
import os.path

import numpy as np

from coex.read import read_lnpi, read_histogram


def activities_to_fractions(activities):
    if len(activities.shape) == 1:
        return np.log(activities)

    fractions = np.copy(activities)
    fractions[0] = np.log(sum(activities))
    fractions[1:] /= np.exp(fractions[0])

    return fractions


def average_histogram(histogram, weights):
    def average_visited_states(states, weight):
        shifted = states['counts'] * np.exp(-weight * states['bins'])

        return sum(shifted * states['bins']) / sum(shifted)

    return np.array([average_visited_states(*pair)
                     for pair in zip(histogram, weights)])


def fractions_to_activities(fractions):
    if len(fractions.shape) == 1:
        return np.exp(fractions)

    activities = np.copy(fractions)
    activity_sum = np.exp(fractions[0])
    activities[1:] *= activity_sum
    activities[0] = activity_sum - sum(activities[1:])

    return activities


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
    # Truncate the first column, which just contains an index, read
    # beta separately, and transpose the rest.
    beta = np.loadtxt(path, usecols=(1, ))
    zz = np.transpose(np.loadtxt(path))[2:]

    return {'beta': beta, 'fractions': zz}


def read_energy_distribution(directory, subensemble):
    hist_file = os.path.join(directory, 'ehist.dat')
    lim_file = os.path.join(directory, 'elim.dat')

    return read_histogram(hist_file, lim_file)[subensemble]


def read_expanded_data(directory, is_tee=False):
    lnpi = read_lnpi(os.path.join(directory, 'lnpi_op.dat'))
    nhists = read_all_molecule_histograms(directory)
    if is_tee:
        bz = read_bz(os.path.join(directory, 'bz.dat'))
        activities = fractions_to_activities(bz['fractions'])

        return {'lnpi': lnpi, 'beta': bz['beta'], 'activities': activities,
                'nhists': nhists}
    else:
        zz = read_zz(os.path.join(directory, 'zz.dat'))
        activities = fractions_to_activities(zz)

        return {'lnpi': lnpi, 'activities': activities, 'nhists': nhists}


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
