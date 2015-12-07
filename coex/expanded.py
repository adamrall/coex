# TODO: Finish writing documentation.
"""Find the coexistence properties of grand canonical expanded
ensemble simulations.
"""

from __future__ import division
import os.path

import numpy as np
import scipy.optimize

from coex.read import read_all_molecule_histograms, read_lnpi


class Phase(object):

    def __init__(self, dist, nhists):
        self.dist = dist
        self.nhists = nhists

    def composition(self, weights):
        nm = self.nmol(weights)

        return nm / sum(nm)

    def grand_potential(self, is_vapor=False, reverse_histogram=False):
        logp = self.dist['logp']
        if not is_vapor:
            return -logp

        gp = np.zeros(len(logp))
        iter_range = range(len(gp))
        if reverse_histogram:
            iter_range = reversed(iter_range)

        for num, i in enumerate(iter_range):
            dist = self.nhists[0][i]
            if dist['bins'][0] < 1.0e-8 and dist['counts'][0] > 1000:
                gp[i] = np.log(dist['counts'][0] / sum(dist['counts']))
            else:
                if num == 0:
                    gp[i] = -self.dist[i]
                else:
                    if reverse_histogram:
                        gp[i] = gp[i + 1] - logp[i + 1] + logp[i]
                    else:
                        gp[i] = gp[i - 1] - logp[i - 1] + logp[i]

        return gp

    def nmol(self, weights):
        return np.array([average_histogram(nh, weights)
                         for nh in self.nhists[1:]])


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


def two_phase_coexistence(first, second, species=1, x0=1.0):
    """Analyze a series of grand canonical expanded ensemble
    simulations.

    Args:
        first: A Phase object with data for the first phase.
        second: A Phase object with data for the second phase.
        species: The integer representing which species to use for the
            reweighting.
        x0: The initial guess to use for the solver in the
            coexistence_point function.

    Returns:
        The coexistence activity ratio, i.e., the quantity
        new_activity / old_activity, for each subensemble.

    Notes:
        The first and second phases must already be shifted to their
        appropriate reference points. See the manual for more
        information.
    """
    first_nh = first.nhists[species]
    second_nh = second.nhists[species]

    def objective(x, j):
        return np.abs(first.dist['logp'][j] + shift_activity(first_nh[j], x) -
                      second.dist['logp'][j] - shift_activity(second_nh[j], x))

    solutions = np.zeros(len(first.dist['logp']))
    for i in range(len(solutions)):
        solutions[i] = scipy.optimize.fsolve(objective, x0=x0, args=(i, ))
        first.dist['logp'][i] += shift_activity(first_nh[i], solutions[i])
        second.dist['logp'][i] += shift_activity(second_nh[i], solutions[i])

    return solutions


def read_phase(directory):
    dist = read_lnpi(os.path.join(directory, 'lnpi_op.dat'))
    nhists = read_all_molecule_histograms(directory)

    return Phase(dist, nhists)


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
