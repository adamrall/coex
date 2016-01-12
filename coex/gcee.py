# gcee.py
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

"""Find the coexistence properties of grand canonical expanded
ensemble simulations.
"""

from __future__ import division
import os.path

import numpy as np
from scipy.optimize import fsolve

from coex.activity import activities_to_fractions, fractions_to_activities
from coex.read import read_bz, read_lnpi, read_zz
from coex.states import average_histogram, read_all_molecule_histograms
from coex.states import read_energy_distribution, reweight_distribution


def get_composition(nhists, weights):
    """Calculate the weighted average composition of a set of
    molecule number visited states histograms.

    Args:
        nhists: The molecule number histograms.
        weights: A list of weights to use for each distribution of
            each histogram.

    Returns:
        A numpy array with the mole fraction of each species in each
        subensemble.
    """
    nm = get_average_n(nhists, weights)

    return nm / sum(nm)


def get_grand_potential(lnpi, nhist, is_vapor=False, is_tee=False):
    """Calculate the grand potential of each subensemble.

    This function walks the length of the expanded ensemble path
    (forwards or backwards) and uses the N=0 visited state
    distribution to calculate the grand potential of each
    subensemble if applicable.  If the N=0 state is not sampled
    sufficiently, the free energy difference between subensembles
    is used.

    Args:
        lnpi: The logarithm of the probability distribution.
        nhist: The total molecule number visited states histogram.
        is_vapor: A boolean denoting whether the phase is a vapor,
            i.e., whether it is likely that the N=0 state is
            sampled.
        is_tee: A boolean denoting whether the simulation uses the
            temperature expanded ensemble. If True, the expanded
            ensemble path is reversed.

    Returns:
        A numpy array with the grand potential of each subensemble.
    """
    if not is_vapor:
        return -lnpi

    gp = np.zeros(len(lnpi))
    iter_range = range(len(gp))
    if is_tee:
        iter_range = reversed(iter_range)

    for num, i in enumerate(iter_range):
        states = nhist[i]
        if states.bins[0] < 1.0e-8 and states.counts[0] > 1000:
            gp[i] = np.log(states.counts[0] / sum(states.counts))
        else:
            if num == 0:
                gp[i] = -lnpi[i]
            else:
                if is_tee:
                    gp[i] = gp[i + 1] - lnpi[i + 1] + lnpi[i]
                else:
                    gp[i] = gp[i - 1] - lnpi[i - 1] + lnpi[i]

    return gp


def get_average_n(nhists, weights):
    """Calculate the weighted average number of molecules.

    Returns:
        A numpy array with the number of molecules of each species
        in each subensemble.
    """
    return np.array([average_histogram(hist, weights[i])
                     for i, hist in enumerate(self.nhists[1:])])


def get_liquid_liquid_coexistence(first, second, species, grand_potential):
    """Find the coexistence point of two liquid phases.

    Args:
        first: The data for the first phase, as returned by
            shift_to_reference().
        second: The data for the second phase.
        species: The species to use for histogram reweighting.
        grand_potential: The reference grand potential.

    Returns:
        A dict with the coexistence logarithm of the probability
        distribution of each phase, the coexistence activity
        fractions, and the histogram weights (for use in finding the
        average N).

    Notes:
        The first and second phases must already be shifted to their
        appropriate reference points. See the prepare_data()
        function.
    """
    fst = first.copy()
    snd = second.copy()
    for p in (fst, snd):
        idx = p['index']
        p['lnpi'] += p['lnpi'][idx] - grand_potential

    return get_two_phase_coexistence(fst, snd, species)


def get_liquid_vapor_coexistence(liquid, vapor, species, is_tee=False):
    """Find the coexistence point of a liquid phase and a vapor
    phase.

    Args:
        liquid: The data for the liquid phase, as returned by
            shift_to_reference().
        vapor: The data for the vapor phase.
        species: The species to use for histogram reweighting.
        is_tee: A bool denoting whether the simulation uses the
            temperature expanded ensemble.

    Returns:
        A dict with the coexistence logarithm of the probability
        distribution of each phase, the coexistence activity
        fractions, and the histogram weights (for use in finding the
        average N).

    Notes:
        The liquid and vapor phases must already be shifted to their
        appropriate reference points. See the prepare_data()
        function.
    """
    liq = liquid.copy()
    vap = vapor.copy()
    vap['lnpi'] = -get_grand_potential(vap['lnpi'], vap['nhists'][0],
                                       is_vapor=True, is_tee=is_tee)
    liq_idx = liq['index']
    vap_idx = vap['index']
    liq['lnpi'] += liq['lnpi'][liq_idx] - vap['lnpi'][vap_idx]

    return get_two_phase_coexistence(liq, vap, species)


def get_two_phase_coexistence(first, second, species=1, x0=1.0):
    """Find the coexistence point of two grand canonical expanded
    ensemble simulations.

    Note that this function is generic: you should use the functions
    get_liquid_vapor_coexistence() or get_liquid_liquid_coexistence()
    instead of using this directly.

    Args:
        first: A Phase object with data for the first phase.
        second: A Phase object with data for the second phase.
        species: The integer representing which species to use for the
            reweighting.
        x0: The initial guess to use for the solver in the
            coexistence_point function.

    Returns:
        A dict with the coexistence logarithm of the probability
        distribution of each phase, the coexistence activity
        fractions, and the histogram weights (for use in finding the
        average N).

    Notes:
        The first and second phases must already be shifted to their
        appropriate reference points.
    """
    first_lnpi = np.copy(first['lnpi'])
    first_nhist = first['nhists'][species]
    second_lnpi = np.copy(second['lnpi'])
    second_nhist = second['nhists'][species]
    init_act = fractions_to_activities(first['fractions'])

    def objective(x, j):
        fst = first['lnpi'][j] + reweight_distribution(first_nhist[j], x)
        snd = second['lnpi'][j] + reweight_distribution(second_nhist[j], x)

        return np.abs(fst - snd)

    for i in range(len(first['lnpi'])):
        if i == first['index'] or i == second['index']:
            continue

        solution = fsolve(objective, x0=x0, args=(i, ))
        first_lnpi[i] += reweight_distribution(first_nhist[i], solution)
        second_lnpi[i] += reweight_distribution(second_nhist[i], solution)

        if species == 0:
            frac = activities_to_fractions(init_act[:, i])
            frac[0] -= solution
            coex_act = fractions_to_activities(frac)
        else:
            new = np.exp(np.log(init_act[species - 1, i]) - solution)
            coex_act[species - 1, i] = new

    return {'first_lnpi': first_lnpi, 'second_lnpi': second_lnpi,
            'fractions': activities_to_fractions(coex_act),
            'weights': np.log(init_act) - np.log(coex_act)}


def read_data(path, is_tee=False):
    """Read the relevant data from an exapnded ensemble simulation
    directory.

    Args:
        path: The directory containing the data.
        is_tee: A bool denoting whether the simulation uses the
            temperature expanded ensemble.

    Returns:
        A dict with the logarithm of the probability distribution,
        activitiy fractions, directory, molecule number visited
        states histograms, and, for TEE simulations, the
        thermodynamic beta (1/kT).
    """
    lnpi = read_lnpi(os.path.join(path, 'lnpi_op.dat'))
    nhists = read_all_molecule_histograms(path)
    if is_tee:
        bb, zz = read_bz(os.path.join(path, 'bz.dat'))

        return {'lnpi': lnpi, 'nhists': nhists, 'beta': bb, 'fractions': zz,
                'path': path}

    zz = read_zz(os.path.join(path, 'zz.dat'))

    return {'lnpi': lnpi, 'nhists': nhists, 'fractions': zz, 'path': path}


def shift_to_reference(data, index, fractions, beta=None):
    """Shift the data to a reference point.

    Args:
        data: A dict, as returned by read_data().
        index: The reference subensemble index.
        fractions: The reference activity fractions.
        beta: The reference thermodynamic beta (1/kT), required only
            for TEE simulations.

    Returns:
        A copy of the data with the shifted logarithm of the
        probability distribution, activity fractions, and reference
        subensemble index.
    """
    res = data.copy()
    res['index'] = index
    if beta is not None:
        energy = read_energy_distribution(res['path'], index)
        diff = beta - res['beta'][index]
        res['beta'][index] = beta
        res['lnpi'][index] += reweight_distribution(energy, diff)

    ref_act = fractions_to_activities(fractions, one_dimensional=True)
    act = fractions_to_activities(res['fractions'])
    ratios = np.log(act[:, index]) - np.log(ref_act)
    act[:, index] = ref_act
    res['fractions'] = activities_to_fractions(act)
    for i, nh in enumerate(res['nhists'][1:]):
        res['lnpi'][index] += reweight_distribution(nh[index], ratios[i])

    return res
