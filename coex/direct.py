"""Analyze direct (grand canonical) simulations."""

from __future__ import division
import os.path

import numpy as np
from scipy.optimize import fsolve

from coex.read import read_lnpi


def transform(distribution, amount):
    """Perform linear transformation on a probability distribution.

    Args:
        distribution: A dict with keys 'param' and 'logp', as
            returned by read_lnpi.
        amount: The amount to shift the distribution using the
            formula ln(P) + N * amount.

    Returns:
        A numpy array with the shifted logarithmic probabilities.
    """
    return distribution['param'] * amount + distribution['logp']


def get_coexistence(directory, activities, species=1):
    """Find the coexistence point of a direct simulation.

    Args:
        directory: The location of the simulation.
        activities: A list of the activities of each species in the
            simulation.
        species: An int denoting which species to use for histogram
            reweighting.

    Returns:
        A dict with the keys 'distribution' and 'activities'
        containing the logarithmic probability distribution and
        activities at the coexistence point.
    """
    dist = read_lnpi(os.path.join(directory, 'lnpi_op.dat'))
    split = int(0.5 * len(dist['param']))

    def objective(x):
        transformed = np.exp(transform(dist, x))
        vapor = sum(transformed[:split])
        liquid = sum(transformed[split:])

        return np.abs(vapor - liquid)

    solution = fsolve(objective, x0=1.0, maxfev=1000)
    dist['logp'] = transform(dist, solution)
    result = {'distribution': dist, 'activities': np.copy(activities)}
    result['activities'][species] *= np.exp(solution)

    return result
