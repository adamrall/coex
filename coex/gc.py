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
        fractions: A 2D numpy array with the activity fractions
           (chi, eta_j) of the simulation.
        weights: The logarithm of the initial activities minus the
            logarithm of the coexistence activities, used to calculate
            the average number of molecules at the coexistence point
            via histogram reweighting.
    """

    def __init__(self, dist, nhists, fractions,
                 weights=None):
        self.dist = dist
        self.nhists = nhists
        self.activities = fractions_to_activities(fractions,
                                                  one_subensemble=True)
        self.weights = weights
        if weights is None:
            self.weights = np.tile(None, len(nhists) - 1)

    def get_composition(self):
        """Calculate the average composition of each phase in the
        simulation.

        Returns:
            A (vapor, liquid) tuple of numpy arrays, each containing the
            mole fraction of each species.
        """
        size = len(self.dist)
        if len(self.molecule_histograms) < 3:
            return np.tile(1.0, size), np.tile(1.0, size)

        vapor_n, liquid_n = self.get_average_n()

        return vapor_n / sum(vapor_n), liquid_n / sum(liquid_n)

    def get_grand_potential(self):
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

    def get_average_n(self, split=0.5):
        """Find the average number of molecules in each phase.

        Args:
            split: A float: where (as a fraction of the order parameter
                range) the liquid/vapor phase boundary lies.

        Returns:
            A (vapor, liquid) tuple of numpy arrays, each containing
            the average number of molecules of each species.
        """
        bound = int(split * len(self.dist))
        nhists = self.molecule_histograms

        def average_phase(hist, weight, phase):
            split_hist = hist[:bound] if phase == 'vapor' else hist[bound:]

            return [d.average(weight) for d in split_hist]

        vapor = np.array([average_phase(nh, self.weights[i], 'vapor')
                          for i, nh in enumerate(nhists[1:])])
        liquid = np.array([average_phase(nh, self.weights[i], 'liquid')
                           for i, nh in enumerate(nhists[1:])])

        return vapor, liquid

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
        dist = self.dist.transform(solution)
        if species == 0:
            frac = activities_to_fractions(self.activities,
                                           one_subensemble=True)
            frac[0] += solution
            act = fractions_to_activities(frac, one_subensemble=True)
        else:
            act = np.copy(self.activities)
            act[species - 1] *= np.exp(solution)
            frac = activities_to_fractions(act, one_subensemble=True)

        weights = np.nan_to_num(np.log(self.activities) - np.log(act))
        nhists = copy.copy(self.molecule_histograms)

        return Simulation(dist=dist, molecule_histograms=nhists,
                          fractions=frac, weights=weights)


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

    return Simulation(dist=dist, nhists=nhists, fractions=fractions)
