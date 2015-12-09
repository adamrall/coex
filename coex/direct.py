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

from coex.read import read_all_molecule_histograms, read_lnpi


class Simulation(object):
    """A container for direct simulation data.

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

    def apply_solutions(self, solutions):
        """Find the coexistence log probability distribution.

        Args:
            solutions: The activity ratios returned by the coexistence
            solver.
        """
        self.dist['logp'] = transform(self.dist, solution)

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
        prob = np.exp(self.dist['logp'])
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
        components = len(self.nhists - 1)
        vapor_n = np.zeros([components])
        liquid_n = np.zeros([components])
        p = np.exp(self.dist['logp'])
        bound = split * len(p)
        vapor_p = p[:bound]
        liquid_p = p[bound:]

        for i, nh in enumerate(self.nhists[1:]):
            nm = sum(nh['counts'] * nh['bins']) / sum(nh['counts'])
            vapor_nm = nm[:bound]
            liquid_nm = nm[bound:]
            vapor_n[i] = sum(vapor_p * vapor_nm / sum(vapor_p))
            liquid_n[i] = sum(liquid_p * liquid_nm / sum(liquid_p))

        return vapor_n, liquid_n


def coexistence(simulation):
    """Find the coexistence point of a direct simulation.

    Args:
        simulation: A simulation object.

    Returns:
        A ratio of the coexistence activity of the order parameter
        species to its initial value.
    """
    split = int(0.5 * len(simulation.dist['param']))

    def objective(x):
        transformed = np.exp(transform(simulation.dist, x))
        vapor = sum(transformed[:split])
        liquid = sum(transformed[split:])

        return np.abs(vapor - liquid)

    return fsolve(objective, x0=1.0, maxfev=1000)


def read_simulation(directory):
    """Read the relevant data from a simulation directory.

    Args:
        directory: The directory containing the data.

    Returns:
        A Simulation object containing the logarithm of the
        probability distribution and the molecule number histograms.
    """
    dist = read_lnpi(os.path.join(directory, 'lnpi_op.dat'))
    nhists = read_all_molecule_histograms(directory)

    return Simulation(dist, nhists)


def transform(dist, amount):
    """Perform linear transformation on a probability distribution.

    Args:
        dist: A dict with keys 'param' and 'logp', as returned by
            read_lnpi.
        amount: The amount to shift the distribution using the
            formula ln(P) + N * amount.

    Returns:
        A numpy array with the shifted logarithmic probabilities.
    """
    return dist['param'] * amount + dist['logp']
