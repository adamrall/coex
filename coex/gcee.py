"""Find the coexistence properties of grand canonical expanded
ensemble simulations.
"""

import copy
import os.path

import numpy as np
from scipy.optimize import fsolve

from coex.activities import activities_to_fractions, fractions_to_activities
from coex.activities import read_activity_fractions
from coex.activities import read_tee_activity_fractions
from coex.distributions import OrderParameterDistribution
from coex.states import read_all_molecule_histograms, VisitedStatesDistribution


class Phase(object):
    def __init__(self, distribution, molecule_histograms, activity_fractions,
                 beta=None, is_vapor=False, weights=None):
        self.distribution = distribution
        self.molecule_histograms = molecule_histograms
        self.activities = fractions_to_activities(activity_fractions)
        self.beta = beta
        self.is_vapor = is_vapor
        self.weights = weights
        self.index = None

    @classmethod
    def from_directory(cls, path, is_vapor=False):
        """Read the relevant data from an exapnded ensemble simulation
        directory.

        Args:
            path: The directory containing the data.
            is_vapor: A boolean denoting whether the phase is a vapor.

        Returns:
            A Phase object with the data contained in the given
            directory.
        """
        dist = OrderParameterDistribution.from_file(
            os.path.join(path, 'lnpi_op.dat'))
        nhists = read_all_molecule_histograms(path)
        beta = None
        try:
            bz = read_tee_activity_fractions(os.path.join(path, 'bz.dat'))
            beta = bz['beta']
            fractions = bz['fractions']
        except FileNotFoundError:
            fractions = read_activity_fractions(os.path.join(path, 'zz.dat'))

        return cls(distribution=dist, molecule_histograms=nhists,
                   activity_fractions=fractions, beta=beta)

    def shift_to_reference(self, index, fractions, beta=None,
                           energy_histogram_path=None):
        """Shift the phase relative to a reference point.

        Args:
            index: The reference subensemble index.
            fractions: The reference activity fractions.
            beta: The reference thermodynamic beta (1/kT), required only
                for TEE simulations.
            energy_histogram_path: The location of the Phase's energy
                histogram, required for TEE simulations.

        Returns:
            A new Phase object shifted to the reference point.
        """
        shifted = copy.copy(self)
        shifted.index = index
        logp = shifted.distribution.log_probabilities
        if beta is not None:
            energy = VisitedStatesDistribution.from_file(energy_histogram_path,
                                                         index)
            diff = beta - self.beta[index]
            shifted.beta[index] = beta
            logp[index] += energy.reweight(diff)

        ref_act = fractions_to_activities(fractions, one_subensemble=True)
        act = self.activities
        ratios = np.nan_to_num(np.log(act[:, index]) - np.log(ref_act))
        shifted.activities[:, index] = ref_act
        for nh, r in zip(shifted.molecule_histograms[1:], ratios):
            logp[index] += nh[index].reweight(r)

        return shifted

    def get_composition(self):
        """Calculate the weighted average composition of the phase.

        Returns:
            A numpy array with the mole fraction of each species in each
            subensemble.
        """
        if len(self.molecule_histograms) < 3:
            return np.tile(1.0, len(self.distribution))

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
        lnpi = self.distribution.log_probabilities
        if not self.is_vapor:
            return -lnpi

        gp = np.zeros(len(lnpi))
        iter_range = range(len(gp))
        nhist = self.molecule_histograms[0]
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
        return np.array([h.average(self.weights[i])
                         for i, h in enumerate(self.molecule_histograms[1:])])


def get_liquid_liquid_coexistence(first, second, species, grand_potential):
    """Find the coexistence point of two liquid phases.

    Args:
        first: A Phase object with data for the first phase.
        second: A Phase object with data for the second phase.
        species: The species to use for histogram reweighting.
        grand_potential: The reference grand potential.

    Returns:
        A tuple with the two Phase objects at coexistence.

    Notes:
        The two phases must already be shifted to their appropriate
        reference points.
    """
    fst = first.copy()
    snd = second.copy()
    for p in (fst, snd):
        idx = p.index
        assert idx is not None, \
            'Phases must be shifted to their reference points.'
        logp = p.distribution.log_probabilities
        logp -= logp[idx] + grand_potential

    return _get_two_phase_coexistence(fst, snd, species)


def get_liquid_vapor_coexistence(liquid, vapor, species):
    """Find the coexistence point of a liquid phase and a vapor
    phase.

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
    liq = liquid.copy()
    vap = vapor.copy()
    assert vap.is_vapor
    vap_logp = vap.distribution.log_probabilities
    liq_logp = liq.distribution.log_probabilities
    vap_logp = -vap.get_grand_potential()
    liq_idx, vap_idx = liq.index, vap.index
    assert liq_idx is not None and vap_idx is not None, \
        'Phases must be shifted to their reference points.'
    liq_logp += vap_logp[vap_idx] - liq_logp[liq_idx]

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
    first_logp = first.distribution.log_probabilities
    first_nhist = first.molecule_histograms[species]
    second_logp = second.distribution.log_probabilities
    second_nhist = second.molecule_histograms[species]
    init_act = np.copy(first.activities)
    if species == 0:
        frac = activities_to_fractions(first.activities)
    else:
        coex_act = np.copy(first.activities)

    def objective(x, j):
        return np.abs(first_logp[j] + first_nhist[j].reweight(x) -
                      second_logp[j] - second_nhist[j].reweight(x))

    for i in range(len(first_logp)):
        if i == first.index or i == second.index:
            continue

        solution = fsolve(objective, x0=x0, args=(i, ))
        first_logp[i] += first_nhist[i].reweight(solution)
        second_logp[i] += second_nhist[i].reweight(solution)

        if species == 0:
            frac[0, i] -= solution
        else:
            coex_act[species - 1, i] = np.exp(
                np.log(init_act[species - 1, i]) - solution)

    if species == 0:
        coex_act = fractions_to_activities(frac)

    first.weights = np.nan_to_num(np.log(init_act) - np.log(coex_act))
    second.weights = np.copy(first.weights)
    first.activities = coex_act
    second.activities = coex_act

    return first, second
