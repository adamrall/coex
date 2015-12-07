# TODO: Finish writing documentation.
"""Find the coexistence properties of direct (grand canonical)
simulations.
"""

from __future__ import division
import os.path

import numpy as np
from scipy.optimize import fsolve

from coex.read import read_all_molecule_histograms, read_lnpi


class Simulation(object):

    def __init__(self, dist, nhists):
        self.dist = dist
        self.nhists = nhists

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

    def composition(self):
        vapor_n, liquid_n = self.nmol()

        return vapor_n / sum(vapor_n), liquid_n / sum(liquid_n)


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

    solution = fsolve(objective, x0=1.0, maxfev=1000)
    simulation.dist['logp'] = transform(simulation.dist, solution)

    return solution


def read_simulation(directory):
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
