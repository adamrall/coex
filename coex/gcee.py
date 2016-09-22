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


class SamplingError(Exception):
    pass


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
            self.weights = np.tile(None, activities.shape)

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
        log_old_act = np.log(self.activities)
        if species == 0:
            frac = activities_to_fractions(self.activities)
            frac[0] -= solutions
            self.activities = fractions_to_activities(frac)

        for i, sol in enumerate(solutions):
            self.dist.log_probs[i] += self.nhists[species][i].reweight(sol)
            if species != 0:
                new_act = np.exp(log_old_act[species - 1, i] - sol)
                self.activities[species - 1, i] = new_act

        self.weights = np.nan_to_num(log_old_act - np.log(self.activities))

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

    @property
    def grand_potential(self):
        """Calculate the grand potential of each subensemble.

        This function walks the length of the expanded ensemble path
        (forwards or backwards) and uses the N=0 visited state
        distribution to calculate the grand potential of each
        subensemble if applicable.  If the N=0 state is not sampled
        sufficiently, the free energy difference between subensembles
        is used.

        Note that this function will not work for liquid phases, which
        do not usually have the N=0 state sampled.

        Returns:
            A numpy array with the grand potential of each subensemble.
        """
        lnpi = self.dist.log_probs
        gp = np.zeros(len(lnpi))
        nhist = self.nhists[0]

        def get_sampled_vapor_subensembles():
            return np.array([True if (d.bins[0] < 1.0e-8 and
                                      d.counts[0] > 1000) else False
                             for d in nhist])

        def calculate_range(iter_range, is_reversed=False):
            in_sampled_block = True
            for i in iter_range:
                d = nhist[i]
                if sampled[i] and in_sampled_block:
                    gp[i] = np.log(d.counts[0] / sum(d.counts))
                else:
                    in_sampled_block = False
                    if is_reversed:
                        gp[i] = gp[i + 1] + lnpi[i + 1] - lnpi[i]
                    else:
                        gp[i] = gp[i - 1] + lnpi[i - 1] - lnpi[i]

        sampled = get_sampled_vapor_subensembles()
        if sampled[0]:
            calculate_range(range(len(gp)))
        elif sampled[-1]:
            calculate_range(reversed(range(len(gp))), is_reversed=True)
        else:
            if np.count_nonzero(sampled) == 0:
                raise SamplingError("{}\n{}".format(
                    "Can't find a sampled subensemble for the grand",
                    'potential calculation. Is this phase a liquid?'))

            first_sampled = np.nonzero(sampled)[0][0]
            calculate_range(range(first_sampled, 0), is_reversed=True)
            calculate_range(range(first_sampled, len(gp)))

        return gp

    def get_average_n(self):
        """Calculate the weighted average number of molecules.

        Returns:
            A numpy array with the number of molecules of each species
            in each subensemble.
        """
        hists = self.nhists[1:]
        if len(hists) > 1:
            return np.array([h.average(w)
                             for h, w in zip(hists, self.weights)])

        return hists[0].average(self.weights[0])


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


def get_liquid_liquid_coexistence(first, second, species, grand_potential,
                                  x0=0.01):
    """Find the coexistence point of two liquid phases.

    Note that the two phases must already be shifted to their
    appropriate reference points.

    Args:
        first: A Phase object with data for the first phase.
        second: A Phase object with data for the second phase.
        species: The species to use for histogram reweighting.
        grand_potential: The reference grand potential.
        x0: The initial guess to use for the solver.

    Returns:
        A tuple with the two Phase objects at coexistence.
    """
    fst = copy.deepcopy(first)
    snd = copy.deepcopy(second)
    for p in fst, snd:
        p.dist.log_propbs -= p.dist[p.index] + grand_potential

    return _get_two_phase_coexistence(fst, snd, species, x0)


def get_liquid_vapor_coexistence(liquid, vapor, species, x0=0.01):
    """Find the coexistence point of a liquid phase and a vapor phase.

    Args:
        liquid: A Phase object with the liquid data.
        vapor: A Phase object with the vapor data.
        species: The species to use for histogram reweighting.
        x0: The initial guess to use for the solver.

    Returns:
        A tuple with the two Phase objects at coexistence.

    Notes:
        The liquid and vapor phases must already be shifted to their
        appropriate reference points.
    """
    liq = copy.deepcopy(liquid)
    vap = copy.deepcopy(vapor)
    try:
        gp = vap.grand_potential
    except SamplingError as e:
        raise RuntimeError('{}\n{}'.format(
            'Consider using get_liquid_liquid_coexistence() with a ',
            'reference grand potential.')) from e

    vap.dist.log_probs = -gp
    liq.dist.log_probs += vap.dist[vap.index] - liq.dist[liq.index]

    return _get_two_phase_coexistence(liq, vap, species, x0)


def _get_two_phase_coexistence(first, second, species, x0):
    """Find the coexistence point of two grand canonical expanded
    ensemble simulations.

    Args:
        first: A Phase object with data for the first phase.
        second: A Phase object with data for the second phase.
        species: The integer representing which species to use for the
            reweighting.
        x0: The initial guess to use for the solver.

    Returns:
        A tuple with the two Phase objects at coexistence.

    Notes:
        The first and second phases must already be shifted to their
        appropriate reference points.
    """
    def solve(i):
        def objective(x):
            fst = first.dist[i] + first.nhists[species][i].reweight(x)
            snd = second.dist[i] + second.nhists[species][i].reweight(x)
            return np.abs(fst - snd)

        if i == first.index or i == second.index:
            return 0.0

        return fsolve(objective, x0=x0)[0]

    solutions = [solve(i) for i in range(len(first.dist))]
    for p in first, second:
        p.shift_to_coexistence(solutions, species)

    return first, second
