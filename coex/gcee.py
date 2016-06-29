"""Find the coexistence properties of grand canonical expanded
ensemble simulations.
"""

import copy
import os.path

import numpy as np
from scipy.optimize import fsolve

from coex.activity import activities_to_fractions, fractions_to_activities
from coex.activity import read_bz, read_zz
from coex.dist import read_lnpi
from coex.hist import read_all_nhists, read_hist


class Phase(object):
    """Calculate the coexistence properties of the output of a grand
    canonical expanded ensemble simulation.

    Attributes:
        dist: A Distribution object.
        path: The location of the simulation data.
        index: The reference subensemble index.
        nhists: A list of molecule number VisitedStatesHistogram
            objects.
        fractions: A numpy array of the (chi, eta_j) activity
            fractions of the simulation.
        beta: An optional list of thermodynamic beta (1/kT) values,
            for temperature expanded ensemble simulations.
        weights: The logarithm of the initial activities minus the
            logarithm of the coexistence activities, used to
            calculate the average number of molecules at the
            coexistence point via histogram reweighting.
    """

    def __init__(self, dist, path, index, nhists=None, activities=None,
                 beta=None, weights=None):
        self.dist = dist
        self.path = path
        self.index = index
        self.nhists = nhists
        self.activities = activities
        self.beta = beta
        self.weights = weights
        if weights is None:
            self.weights = np.tile(None, len(dist))

    @property
    def fractions(self):
        return activities_to_fractions(self.activities)

    @fractions.setter
    def fractions(self, frac):
        self.activities = fractions_to_activities(frac)

    def shift_to_coexistence(self, solutions, species):
        """Shift the activities and order parameter probability
        distribution to the coexistence point.

        Args:
            solutions: A list of log(activitiy) differences.
            species: The species used in histogram reweighting.
        """
        if species == 0:
            frac = activities_to_fractions(self.activities)
            frac[0] -= solutions
            self.activities = fractions_to_activities(frac)

        for i, sol in enumerate(solutions):
            self.dist.log_probs[i] += self.nhists[species][i].reweight(sol)
            if species != 0:
                log_old_act = np.log(self.activities[species - 1, i])
                log_new_act = log_old_act - sol
                self.weights[i] = np.nan_to_num(log_old_act - log_new_act)
                self.activities[species - 1, i] = np.exp(log_new_act)

    def get_composition(self):
        """Calculate the weighted average composition of the phase.

        Returns:
            A numpy array with the mole fraction of each species in each
            subensemble.
        """
        if len(self.nhists) < 3:
            return np.tile(1.0, len(self.dist))

        nm = self.get_average_n()

        return nm / sum(nm)

    def get_grand_potential(self):
        """Calculate the grand potential of each subensemble.

        This function walks the length of the expanded ensemble path
        (forwards or backwards) and uses the N=0 visited state
        distribution to calculate the grand potential of each
        subensemble if applicable.  If the N=0 state is not sampled
        sufficiently, the free energy difference between subensembles
        is used.

        Returns:
            A numpy array with the grand potential of each subensemble.
        """
        lnpi = self.dist.log_probs
        gp = np.zeros(len(lnpi))
        iter_range = range(len(gp))
        nhist = self.nhists[0]
        # Reverse the order of traversal for TEE: the subensembles
        # more likely to sample N=0 are at high beta (low T).
        if self.beta is not None:
            iter_range = reversed(iter_range)

        for num, i in enumerate(iter_range):
            d = nhist[i]
            if d.bins[0] < 1.0e-8 and d.counts[0] > 1000:
                gp[i] = np.log(d.counts[0] / sum(d.counts))
            else:
                if num == 0:
                    gp[i] = -lnpi[i]
                else:
                    if self.beta is not None:
                        gp[i] = gp[i + 1] + lnpi[i + 1] - lnpi[i]
                    else:
                        gp[i] = gp[i - 1] + lnpi[i - 1] - lnpi[i]

        return gp

    def get_average_n(self):
        """Calculate the weighted average number of molecules.

        Returns:
            A numpy array with the number of molecules of each species
            in each subensemble.
        """
        return np.array([h.average(w)
                         for h, w in zip(self.nhists[1:], self.weights)])


def read_phase(path, index, fractions=None, beta=None):
    """Read the relevant data from an exapnded ensemble simulation
    directory.

    Args:
        path: The directory containing the data.
        index: The reference subensemble index.
        fractions: The reference activity fractions.
        beta: The reference inverse temperature (1/kT).

    Returns:
        A Phase object with the data contained in the given directory.
    """
    dist = read_lnpi(os.path.join(path, 'lnpi_op.dat'))
    nhists = read_all_nhists(path)
    bb = None
    try:
        bz = read_bz(os.path.join(path, 'bz.dat'))
        bb = bz['beta']
        act = fractions_to_activities(bz['fractions'])
    except FileNotFoundError:
        try:
            act = fractions_to_activities(
                read_zz(os.path.join(path, 'zz.dat')))
        except FileNotFoundError:
            act = None

    logp_shift = -dist.log_probs[index]
    if beta is not None:
        energy = read_hist(os.path.join(path, 'ehist.dat'))[index]
        diff = beta - bb[index]
        bb[index] = beta
        logp_shift += energy.reweight(diff)

    if fractions is not None:
        ref_act = fractions_to_activities(fractions, one_subensemble=True)
        ratios = np.nan_to_num(np.log(act[:, index]) - np.log(ref_act))
        act[:, index] = ref_act
        for nh, r in zip(nhists[1:], ratios):
            logp_shift += nh[index].reweight(r)

    dist.log_probs += logp_shift

    return Phase(dist=dist, path=path, index=index, nhists=nhists,
                 activities=act, beta=bb)


def read_phases(paths, index, fractions=None, beta=None):
    """Read the relevant data from a list of exapnded ensemble
    simulations with the same reference point.

    Args:
        paths: A list of directories containing the data to read.
        index: The reference subensemble index.
        fractions: The reference activity fractions.
        beta: The reference inverse temperature (1/kT).

    Returns:
        A tuple of Phases the data in each directory.
    """
    return (read_phase(p, index, fractions, beta) for p in paths)


def get_liquid_liquid_coexistence(first, second, species, grand_potential):
    """Find the coexistence point of two liquid phases.

    Note that the two phases must already be shifted to their
    appropriate reference points.

    Args:
        first: A Phase object with data for the first phase.
        second: A Phase object with data for the second phase.
        species: The species to use for histogram reweighting.
        grand_potential: The reference grand potential.

    Returns:
        A tuple with the two Phase objects at coexistence.
    """
    fst = copy.deepcopy(first)
    snd = copy.deepcopy(second)
    for p in fst, snd:
        p.dist.log_propbs -= p.dist[p.index] + grand_potential

    return _get_two_phase_coexistence(fst, snd, species)


def get_liquid_vapor_coexistence(liquid, vapor, species):
    """Find the coexistence point of a liquid phase and a vapor phase.

    Args:
        liquid: A Phase object with the liquid data.
        vapor: A Phase object with the vapor data.
        species: The species to use for histogram reweighting.

    Returns:
        A tuple with the two Phase objects at coexistence.

    Notes:
        The liquid and vapor phases must already be shifted to their
        appropriate reference points.
    """
    liq = copy.deepcopy(liquid)
    vap = copy.deepcopy(vapor)
    vap.dist.log_probs = -vap.get_grand_potential()
    liq.dist.log_probs += vap.dist[vap.index] - liq.dist[liq.index]

    return _get_two_phase_coexistence(liq, vap, species)


def _get_two_phase_coexistence(first, second, species=1, x0=0.01):
    """Find the coexistence point of two grand canonical expanded
    ensemble simulations.

    Args:
        first: A Phase object with data for the first phase.
        second: A Phase object with data for the second phase.
        species: The integer representing which species to use for the
            reweighting.
        x0: The initial guess to use for the solver in the
            coexistence_point function.

    Returns:
        A tuple with the two Phase objects at coexistence.

    Notes:
        The first and second phases must already be shifted to their
        appropriate reference points.
    """
    def objective(x, j):
        if j == first.index or j == second.index:
            return 0.0

        return np.abs(first.dist[j] + first.nhists[species][j].reweight(x) -
                      second.dist[j] - second.nhists[species][j].reweight(x))

    solutions = [fsolve(objective, x0=x0, args=(i, ))
                 for i in range(len(first.dist))]
    for p in first, second:
        p.shift_to_coexistence(solutions, species)

    return first, second
