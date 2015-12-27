# TODO: finish updating documentation
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
from scipy.optimize import fsolve

from coex.activity import activities_to_fractions, fractions_to_activities
from coex.read import read_bz, read_lnpi, read_zz
from coex.states import average_histogram, read_all_molecule_histograms
from coex.states import read_energy_distribution


class Phase(object):
    """A container for grand canonical exapnded ensemble simulation data.

    Attributes:
        lnpi: A numpy array with the logarithm of the probability
            distribution.
        nhists: A list of molecule number histograms.

    See Also:
        read.histogram() for a description of the structure of each
        histogram.
    """

    def __init__(self, lnpi, nhists, index, activities, beta=None):
        self.lnpi = lnpi
        self.nhists = nhists
        self.index = index
        self.init_act = activities
        self.coex_act = np.copy(activities)
        self.beta = beta

    def composition(self):
        """Calculate the weighted average composition of the phase.

        Returns:
            A numpy array with the mole fraction of each species in
            each subensemble.
        """
        nm = self.nmol()

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
        if not is_vapor:
            return -self.lnpi

        gp = np.zeros(len(self.lnpi))
        iter_range = range(len(gp))
        if reverse_histogram:
            iter_range = reversed(iter_range)

        for num, i in enumerate(iter_range):
            states = self.nhists[0][i]
            if states.bins[0] < 1.0e-8 and states.counts[0] > 1000:
                gp[i] = np.log(states.counts[0] / sum(states.counts))
            else:
                if num == 0:
                    gp[i] = -self.lnpi[i]
                else:
                    if reverse_histogram:
                        gp[i] = gp[i + 1] - self.lnpi[i + 1] + self.lnpi[i]
                    else:
                        gp[i] = gp[i - 1] - self.lnpi[i - 1] + self.lnpi[i]

        return gp

    def nmol(self):
        """Calculate the weighted average number of molecules in the
        phase.

        Returns:
            A numpy array with the number of molecules of each species
            in each subensemble.
        """
        weights = np.log(self.coex_act) - np.log(self.init_act)

        return np.array([average_histogram(hist, weights[i])
                         for i, hist in enumerate(self.nhists[1:])])


def liquid_liquid_coexistence(first, second, species, grand_potential):
    for p in (first, second):
        p.lnpi += p.lnpi[p.index] - grand_potential

    two_phase_coexistence(first, second, species)


def liquid_vapor_coexistence(liquid, vapor, species):
    vapor.lnpi = -vapor.grand_potential()
    liquid.lnpi += liquid.lnpi[liquid.index] - vapor.lnpi[vapor.index]
    two_phase_coexistence(liquid, vapor, species)


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

    Notes:
        The first and second phases must already be shifted to their
        appropriate reference points.
    """
    first_nhist = first.nhists[species]
    second_nhist = second.nhists[species]

    def objective(x, j):
        return np.abs(first.lnpi[j] + first_nhist[j].reweight(x) -
                      second.lnpi[j] - second_nhist[j].reweight(x))

    for i in range(len(first.lnpi)):
        if i == first.index or i == second.index:
            continue

        solution = fsolve(objective, x0=x0, args=(i, ))
        first.lnpi[i] += first_nhist[i].reweight(solution)
        second.lnpi[i] += second_nhist[i].reweight(solution)

        if species == 0:
            frac = activities_to_fractions(first.init_act[:, i])
            frac[0] -= solution
            act = fractions_to_activities(frac)
            first.coex_act[:, i] = act
            second.coex_act[:, i] = act
        else:
            new = np.exp(np.log(first.init_act[species - 1, i]) - solution)
            first.coex_act[species - 1, i] = new
            second.coex_act[species - 1, i] = new


def read_phase(directory, index, fractions, beta=None):
    """Read the relevant data from an exapnded ensemble simulation
    directory.

    Args:
        directory: The directory containing the data.
        index: The reference subensemble index.
        fractions: The reference activity fractions.
        beta: A numpy array with the reference thermodynamic beta.

    Returns:
        A Phase object, shifted to the refrence point.
    """
    lnpi = read_lnpi(os.path.join(directory, 'lnpi_op.dat'))
    nhists = read_all_molecule_histograms(directory)
    bb = None
    if beta is not None:
        bb, zz = read_bz(os.path.join(directory, 'bz.dat'))
        energy = read_energy_distribution(directory, index)
        lnpi[index] += energy.reweight(beta - bb[index])

    zz = read_zz(os.path.join(directory, 'zz.dat'))
    ref_act = fractions_to_activities(zz, direct=True)
    act = fractions_to_activities(fractions)
    ratios = np.log(act[:, index]) - np.log(ref_act)
    act[:, index] = ref_act
    for i, nh in enumerate(nhists[1:]):
        lnpi[index] += nh[index].reweight(ratios[i])

    return Phase(lnpi, nhists, act, beta=bb)
