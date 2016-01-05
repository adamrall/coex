# TODO: finish updating documentation
# gc.py
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


def get_composition(nhists, weight):
    """Calculate the average composition of each phase.

    Args:
        nhists: The molecule number visited states histograms.
        weight: The logarithm of the initial order parameter activity
            minus the logarithm of the coexistence activity.

    Returns:
        A (vapor, liquid) tuple of numpy arrays, each containing
        the mole fraction of each species.
    """
    vapor_n, liquid_n = get_average_n(nhists, weight)

    return vapor_n / sum(vapor_n), liquid_n / sum(liquid_n)


def get_grand_potential(lnpi):
    """Calculate the grand potential given the logarithm of the
    probability distribution.

    If the order parameter is the total molecule number N, then
    this is the absolute grand potential of the system.
    Otherwise, it is a relative value for the analyzed species.

    Args:
        lnpi: The logarithm of the probability distribution.

    Returns:
        A float containing the grand potential.
    """
    prob = np.exp(lnpi)
    prob /= sum(prob)

    return np.log(prob[0] * 2.0)


def get_average_n(nhists, weight, split=0.5):
    """Find the average number of molecules in each phase.

    Args:
        nhists: The molecule number visited states histograms.
        weight: The logarithm of the initial order parameter activity
            minus the logarithm of the coexistence activity.
        split: A float: where (as a fraction of the order parameter
            range) the liquid/vapor phase boundary lies.

    Returns:
        A (vapor, liquid) tuple of numpy arrays, each containing
        the average number of molecules of each species.
    """
    bound = split * len(nhists[1])
    vapor = np.array([average_histogram(nh[:bound], weight)
                      for nh in nhists[1:]])
    liquid = np.array([average_histogram(nh[bound:], weight)
                       for nh in nhists[1:]])

    return vapor, liquid


def transform(order_param, lnpi, amount):
    """Perform linear transformation on a probability distribution.

    Args:
        order_param: The order parameter values.
        lnpi: The logarithm of the probabilities.
        amount: The amount to shift the distribution.

    Returns:
        A numpy array with the shifted logarithmic probabilities.
    """
    return order_param * amount + lnpi


def get_coexistence(sim, fractions, species):
    """Find the coexistence point of a direct simulation.

    Args:
        sim: A dict, as returned by read_simulation().
        fractions: A numpy array with the simulation's activity
            fractions.
        species: The simulation's order parmeter species.
    """
    init_act = fractions_to_activities(fractions)
    split = int(0.5 * len(sim['order_param']))

    def objective(x):
        transformed = np.exp(transform(sim['order_param'], sim['lnpi'], x))
        vapor = sum(transformed[:split])
        liquid = sum(transformed[split:])

        return np.abs(vapor - liquid)

    solution = fsolve(objective, x0=1.0, maxfev=1000)
    lnpi = transform(sim['order_param'], sim['lnpi'], solution)
    if species == 0:
        frac = activities_to_fractions(init_act, one_dimensional=True)
        frac[0] += solution
        coex_act = fractions_to_activities(frac, one_dimensional=True)
    else:
        coex_act[species - 1] *= np.exp(solution)

    return {'lnpi': lnpi, 'fractions': activities_to_fractions(coex_act),
            'weight': np.log(init_act) - np.log(coex_act)}


def read_simulation(directory):
    """Read the relevant data from a simulation directory.

    Args:
        directory: The directory containing the data.

    Returns:
        A dict with the order parameter, logarithm of the
        probability distribution, and molecule number visited states
        histograms.
    """
    lnpi_file = os.path.join(directory, 'lnpi_op.dat')
    order_param = read_order_parameter(lnpi_file)
    lnpi = read_lnpi(lnpi_file)
    nhists = read_all_molecule_histograms(directory)

    return {'order_param': order_param, 'lnpi': lnpi, 'nhists': nhists}
