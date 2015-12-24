# TODO: finish updating documentation
# direct.py
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

"""Find the coexistence properties of direct (grand canonical)
simulations.
"""

from __future__ import division
import os.path

import numpy as np
from scipy.optimize import fsolve

from coex.activity import activities_to_fractions, fractions_to_activities
from coex.read import read_order_parameter, read_lnpi
from coex.states import average_histogram, read_all_molecule_histograms


class Simulation(object):
    """A container for direct simulation data.

    Attributes:
        order_param: A numpy array with the values of the order
            parameter (typically the number of molecules of one or
            all species).
        lnpi: A numpy array with the logarithm of the probability
            distribution.
        nhists: A list of molecule number histograms.

    See Also:
        read.histogram() for a description of the structure of each
        histogram.
    """

    def __init__(self, order_param, lnpi, nhists, activities):
        self.order_param = order_param
        self.lnpi = lnpi
        self.nhists = nhists
        self.init_act = activities
        self.coex_act = np.copy(activities)

    def composition(self):
        """Calculate the average composition of each phase.

        Returns:
            A (vapor, liquid) tuple of numpy arrays, each containing
            the mole fraction of each species.
        """
        vapor_n, liquid_n = self.nmol()

        return vapor_n / sum(vapor_n), liquid_n / sum(liquid_n)

    def grand_potential(self):
        """Calculate the grand potential of the simulation.

        If the order parameter is the total molecule number N, then
        this is the absolute grand potential of the system.
        Otherwise, it is a relative value for the analyzed species.

        Returns:
            A float containing the grand potential.
        """
        prob = np.exp(self.lnpi)
        prob /= sum(prob)

        return np.log(prob[0] * 2.0)

    def nmol(self, split=0.5):
        """Find the average number of molecules in each phase.

        Args:
            split: A float: where (as a fraction of the order
                parameter range) the liquid/vapor phase boundary lies.

        Returns:
            A (vapor, liquid) tuple of numpy arrays, each containing
            the average number of molecules of each species.
        """
        weight = np.log(self.init_act) - np.log(self.coex_act)
        bound = split * len(self.lnpi)
        vapor = np.array([average_histogram(nh[:bound], weight)
                          for nh in self.nhists[1:]])
        liquid = np.array([average_histogram(nh[bound:], weight)
                           for nh in self.nhists[1:]])

        return vapor, liquid

    def transform(self, amount):
        """Perform linear transformation on a probability distribution.

        Args:
            amount: The amount to shift the distribution using the
                formula ln(P) + N * amount.

        Returns:
            A numpy array with the shifted logarithmic probabilities.
        """
        return self.order_param * amount + self.lnpi


def coexistence(sim, species):
    """Find the coexistence point of a direct simulation.

    Args:
        sim: A simulation object.
        species: The simulation's order parmeter species.
    """
    split = int(0.5 * len(sim.order_param))

    def objective(x):
        transformed = np.exp(sim.transform(x))
        vapor = sum(transformed[:split])
        liquid = sum(transformed[split:])

        return np.abs(vapor - liquid)

    solution = fsolve(objective, x0=1.0, maxfev=1000)
    sim.lnpi = sim.transform(solution)
    if species == 0:
        frac = activities_to_fractions(sim.init_act, direct=True)
        frac[0] += solution
        sim.coex_act = fractions_to_activities(frac, direct=True)
    else:
        sim.coex_act[species - 1] *= np.exp(solution)


def read_simulation(directory, fractions):
    """Read the relevant data from a simulation directory.

    Args:
        directory: The directory containing the data.

    Returns:
        A Simulation object containing the logarithm of the
        probability distribution and the molecule number histograms.
    """
    lnpi_file = os.path.join(directory, 'lnpi_op.dat')
    order_param = read_order_parameter(lnpi_file)
    lnpi = read_lnpi(lnpi_file)
    nhists = read_all_molecule_histograms(directory)
    activities = fractions_to_activities(fractions, direct=True)

    return Simulation(order_param, lnpi, nhists, activities)
