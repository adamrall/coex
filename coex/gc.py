"""Find the coexistence properties of direct (grand canonical)
simulations.
"""

import copy
import os.path

import numpy as np
from scipy.optimize import fsolve

from coex.activity import activities_to_fractions, fractions_to_activities
from coex.dist import read_lnpi
from coex.hist import read_all_nhists


class Simulation(object):
    """Calculate the coexistence properties of the output from a direct
    grand canonical simulation.

    Attributes:
        dist: An OrderParameterDistribution object.
        nhists: A list of molecule number VisitedStatesHistogram
            objects.
        activities: A numpy array with the activities of each species.
        weights: The logarithm of the initial activities minus the
            logarithm of the coexistence activities, used to calculate
            the average number of molecules at the coexistence point
            via histogram reweighting.
    """

    def __init__(self, dist, nhists, activities, weights=None):
        self.dist = dist
        self.nhists = nhists
        self.activities = activities
        self.weights = weights
        if weights is None:
            self.weights = np.tile(None, len(nhists) - 1)

    @property
    def fractions(self):
        return np.reshape(activities_to_fractions(self.activities,
                                                  one_subensemble=True),
                          len(self.activities))

    @fractions.setter
    def fractions(self, frac):
        if not isinstance(frac, np.ndarray):
            transposed = np.transpose(np.array([frac]))
        else:
            transposed = np.transpose(frac)

        self.activities = fractions_to_activities(transposed,
                                                  one_subensemble=True)

    @property
    def composition(self):
        """Calculate the average composition of each phase in the
        simulation.

        Returns:
            A (vapor, liquid) tuple of numpy arrays, each containing the
            mole fraction of each species.
        """
        size = len(self.dist)
        if len(self.nhists) < 3:
            return {'vapor': np.tile(1.0, size), 'liquid': np.tile(1.0, size)}

        avg_n = self.average_n

        return {'vapor': avg_n['vapor'] / sum(avg_n['vapor']),
                'liquid': avg_n['liquid'] / sum(avg_n['liquid'])}

    @property
    def grand_potential(self):
        """Calculate the grand potential of the Simulation.

        If the order parameter is the total molecule number N, then
        this is the absolute grand potential of the system.
        Otherwise, it is a relative value for the analyzed species.

        Returns:
            A float containing the grand potential.
        """
        prob = np.exp(self.dist.log_probs)
        prob /= sum(prob)

        return np.log(prob[0] * 2.0)

    @property
    def average_n(self, split=0.5):
        """Find the average number of molecules in each phase.

        Args:
            split: A float: where (as a fraction of the order parameter
                range) the liquid/vapor phase boundary lies.

        Returns:
            A (vapor, liquid) tuple of numpy arrays, each containing
            the average number of molecules of each species.
        """
        bound = int(split * len(self.dist))

        def average_phase(hist, weight, is_vapor=True):
            split_hist = hist[:bound] if is_vapor else hist[bound:]

            return [d.average(weight) for d in split_hist]

        hists = self.nhists[1:]
        if len(hists) > 1:
            vapor = np.array([average_phase(nh, self.weights[i])
                              for i, nh in enumerate(hists)])
            liquid = np.array(
                [average_phase(nh, self.weights[i], is_vapor=False)
                 for i, nh in enumerate(hists)])
        else:
            vapor = average_phase(hists[0], self.weights[0])
            liquid = average_phase(hists[0], self.weights[0], is_vapor=False)

        return {'vapor': vapor, 'liquid': liquid}

    def get_coexistence(self, species=1, x0=-0.001):
        """Find the coexistence point of the simulation.

        Args:
            species: The simulation's order parmeter species.
            x0: The initial guess for the optimization solver.

        Returns:
            A new Simulation object at the coexistence point.
        """
        def objective(x):
            vapor, liquid = self.dist.split()

            return np.abs(sum(np.exp(vapor.log_probs + x * vapor.index)) -
                          sum(np.exp(liquid.log_probs + x * liquid.index)))

        solution = fsolve(objective, x0=x0, maxfev=10000)
        coex = copy.deepcopy(self)
        coex.dist.log_probs += solution * coex.dist.index
        if species == 0:
            frac = activities_to_fractions(self.activities,
                                           one_subensemble=True)
            frac[0] += solution
            coex.activities = fractions_to_activities(frac,
                                                      one_subensemble=True)
        else:
            coex.activities[species - 1] *= np.exp(solution)

        coex.weights = np.nan_to_num(np.log(self.activities) -
                                     np.log(coex.activities))

        return coex


def get_coexistence(sim, species, x0=-0.001):
    """Find the coexistence point of the simulation.

    Args:
        species: The simulation's order parmeter species.
        x0: The initial guess for the optimization solver.

    Returns:
        A new Simulation object at the coexistence point.
    """
    return sim.get_coexistence(species, x0)


def read_simulation(path, fractions):
    """Read the relevant data from a simulation directory.

    Args:
        path: The directory containing the data.
        fractions: The activity fractions (chi, eta_j) of the
            simulation.

    Returns:
        A dict with the order parameter, logarithm of the
        probability distribution, and molecule number visited
        states histograms.
    """
    dist = read_lnpi(os.path.join(path, 'lnpi_op.dat'))
    nhists = read_all_nhists(path)
    if not isinstance(fractions, np.ndarray):
        frac = np.transpose(np.array([fractions]))
    else:
        frac = np.transpose(fractions)

    act = fractions_to_activities(frac)

    return Simulation(dist=dist, nhists=nhists, activities=act)
