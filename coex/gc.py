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


def get_average_n(nhists, weights, split=0.5):
    """Find the average number of molecules in each phase.

    Args:
        nhists: The molecule number visited states histograms.
        weights: The logarithm of the initial activities minus the
            logarithm of the coexistence activities.
        split: A float: where (as a fraction of the order parameter
            range) the liquid/vapor phase boundary lies.

    Returns:
        A (vapor, liquid) tuple of numpy arrays, each containing
        the average number of molecules of each species.
    """
    size = len(nhists[1])
    bound = split * size
    vapor = np.array([average_histogram(nh[:bound], np.tile(weights[i], size))
                      for i, nh in enumerate(nhists[1:])])
    liquid = np.array([average_histogram(nh[bound:], np.tile(weights[i], size))
                       for i, nh in enumerate(nhists[1:])])

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


def get_coexistence(sim, fractions, species, x0=-0.001):
    """Find the coexistence point of a direct simulation.

    Args:
        sim: A dict, as returned by read_simulation().
        fractions: A numpy array with the simulation's activity
            fractions.
        species: The simulation's order parmeter species.
        x0: The initial guess for the optimization solver.

    Returns:
        A dict with the coexistence logarithm of the probability
        distribution, activity fractions, and histogram weights (for
        use in finding the weighted average of N).
    """
    init_act = fractions_to_activities(fractions)
    split = int(0.5 * len(sim['order_param']))

    def objective(x):
        transformed = np.exp(transform(sim['order_param'], sim['lnpi'], x))
        vapor = sum(transformed[:split])
        liquid = sum(transformed[split:])

        return np.abs(vapor - liquid)

    solution = fsolve(objective, x0=x0, maxfev=10000)
    lnpi = transform(sim['order_param'], sim['lnpi'], solution)
    if species == 0:
        frac = activities_to_fractions(init_act, one_subensemble=True)
        frac[0] += solution
        coex_act = fractions_to_activities(frac, one_subensemble=True)
    else:
        coex_act = np.copy(init_act)
        coex_act[species - 1] *= np.exp(solution)

    weight = np.nan_to_num(np.log(init_act) - np.log(coex_act))

    return {'lnpi': lnpi, 'fractions': activities_to_fractions(coex_act),
            'weight': weight}


def read_data(path):
    """Read the relevant data from a simulation directory.

    Args:
        path: The directory containing the data.

    Returns:
        A dict with the order parameter, logarithm of the
        probability distribution, and molecule number visited states
        histograms.
    """
    lnpi_file = os.path.join(path, 'lnpi_op.dat')
    order_param = read_order_parameter(lnpi_file)
    lnpi = read_lnpi(lnpi_file)
    nhists = read_all_molecule_histograms(path)

    return {'order_param': order_param, 'lnpi': lnpi, 'nhists': nhists}
