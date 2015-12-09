# expanded.py
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

"""Find the coexistence properties of grand canonical expanded
ensemble simulations.
"""

from __future__ import division
import os.path

import numpy as np
import scipy.optimize

from coex.read import read_all_molecule_histograms, read_lnpi


class Phase(object):
    """A container for grand canonical exapnded ensemble simulation data.

    Attributes:
        dist: A dict with the keys 'param' and 'logp' holding the
            logarithm of the probability distribution.
        nhists: A list of molecule number histograms.

    See Also:
        read.histogram() for a description of the structure of each
        histogram.
    """

    def __init__(self, dist, nhists):
        self.dist = dist
        self.nhists = nhists

    def apply_solutions(self, solutions, species=1):
        """Find the coexistence log probability distribution.

        Apply the activity ratios calculated by the coexistence
        solvers to the dist attribute.

        Args:
            solutions: A list of activity ratios.

            species: The chemical species for which the solutions
            apply.  May be 0 if solutions corresponds to the ratios
            for the sum of the activities.
        """
        for i, nd in enumerate(self.nhists[species]):
            self.dist['logp'][i] += shift_activity(nd, solutions[i])

    def composition(self, weights=None):
        """Calculate the weighted average composition of the phase.

        The weights here are frequently the solutions found by one of
        the coexistence point solving functions in this module.  They
        correspond to ratios of new/old activities for a given order
        parameter species.

        Args:
            weights: A numpy array with weights for each species in
                each subensemble.

        Returns:
            A numpy array with the mole fraction of each species in
            each subensemble.

        See Also:
            solutions_to_weights() for a function to convert the
            output of the coexistence functions into a form suitable
            for use here.
        """
        nm = self.nmol(weights)

        return nm / sum(nm)

    def grand_potential(self, is_vapor=False, reverse_histogram=False):
        """Calculate the grand potential of each subensemble in the
        phase.

        This function walks the length of the expanded ensemble path
        (forwards or backwards) and uses the N=0 visited state
        distribution to calculate the grand potential of each
        subensemble if applicable.  If the N=0 state is not sampled
        sufficiently, the free energy difference between subensembles
        is used.

        Args:
            is_vapor: A boolean denoting whether the phase is a vapor,
                i.e., whether it is likely that the N=0 state is
                sampled.
            reverse_histogram: A boolean denoting which direction to
                traverse the expanded ensemble path.  Should be True
                for TEE simulations.

        Returns:
            A numpy array with the grand potential of each
            subensemble.
        """
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

    def nmol(self, weights=None):
        """Calculate the weighted average number of molecules in the
        phase.

        Args:
            weights: A numpy array with weights for each species in
                each subensemble.

        Returns:
            A numpy array with the number of molecules of each species
            in each subensemble.

        See Also:
            composition() and solutions_to_weights() for descriptions
            of the weights used here.
        """
        return np.array([average_histogram(nh, weights[i])
                         for i, nh in enumerate(self.nhists[1:])])


def activities_to_fractions(activities):
    """Convert a list of activities to activity fractions.

    Args:
        activities: A numpy array with the activities of the system.
            Each row corresponds to a species and each column to a
            subensemble.

    Returns:
        A numpy array.  The first row contains the logarithm of the
        sum of the activities in each subensemble.  The subsequent
        rows contain the activity fractions of each species after the
        first for each subensemble.

    See Also:
        fractions_to_activities() for the opposite conversion.
    """
    if len(activities.shape) == 1:
        return np.log(activities)

    fractions = np.copy(activities)
    fractions[0] = np.log(sum(activities))
    fractions[1:] /= np.exp(fractions[0])

    return fractions


def average_histogram(histogram, weights=None):
    """Calculate the weighted average of a visited states histogram.

    Args:
        histogram: A vistied states histogram.
        weights: A list of weights for each distribution in the
            histogram.

    Returns:
        A numpy array with the weighted average of each distribution
        in the histogram.

    See Also:
        read.read_histogram() for a description of the structure of
        the histogram.
    """

    def average_visited_states(states, weight):
        """Average a given visited states distribution."""
        shifted = states['counts'] * np.exp(-weight * states['bins'])

        return sum(shifted * states['bins']) / sum(shifted)

    if weights is None:
        weights = np.ones(len(histogram))

    return np.array([average_visited_states(*pair)
                     for pair in zip(histogram, weights)])


def fractions_to_activities(fractions):
    """Convert a list of activity fractions to activities.

    Args:
        fractions: A numpy array with the activity fractions.  The
            first row is the log of the sum of activities; each
            subsequent row is the activity fraction of the 2nd, 3rd,
            etc. species.  Each column is a subensemble.

    Returns:
        A numpy array with the activities: each row is a species, each
        column a subensemble.

    See Also:
        activities_to_fractions() for the opposite conversion.
    """
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

    return [scipy.optimize.fsolve(objective, x0=x0, args=(i, ))
            for i in range(len(first.dist['logp']))]
    

def read_phase(directory):
    """Read the relevant data from an exapnded ensemble simulation
    directory.

    Args:
        directory: The directory containing the data.

    Returns:
        A Phase object containing the logarithm of the probability
        distribution and the molecule number histograms.
    """
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


def solutions_to_weights(solutions, species_count=1, order_parameter=1,
                         fractions=None):
    """Converts the solutions returned by the coexistence finding
    functions into a form used for finding the average molecule number
    and composition of a phase.

    Args:
        solutions: A numpy array of activity ratios.
        species_count: The number of species in the simulation.
        order_parameter: The order parameter species, i.e., the
            species represented by the ratios in the solutions.  Set
            to 0 if the sum of activities was used as the order
            parameter.
        fractions: Required only for order_parameter=0; a list of the
            activity fractions of the simulation.

    Returns:
        A numpy array with the appropriate weights to use for
        averaging the molecule number histograms of a given Phase
        object.

    See Also:
        Phase.composition(), Phase.nmol()

    """
    if order_parameter == 0:
        old_activities = fractions_to_activities(fractions)
        new_fractions = np.copy(fractions)
        new_fractions[0] += np.log(solutions)
        new_activities = fractions_to_activities(new_fractions)

        return new_activities / old_activities

    if species_count == 1:
        return solutions

    weights = np.ones([species_count, len(solutions)])
    if order_parameter > 0:
        weights[order_parameter] = solutions

    return weights
