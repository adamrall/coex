"""Functions for smoothing lnpi_op and lnpi_tr."""

import os.path

import numpy as np


def find_poorly_sampled(transitions, cutoff):
    """Determine which subensemble/molecule/growth stage combinations
    are not adequately sampled.

    For each combination, we take the minimum of the number of
    forward and backward transitions. If this number is less than the
    average over all combinations times some cutoff fraction, then we
    add it to the list of poorly sampled combinations.

    Args:
        transitions: A 2D numpy array with the count of forward and
            backward transitions for each subensemble.
        cutoff: The fraction of the mean to use as a threshold for
            sampling quality.

    Returns:
        A list of indices which don't meet the sampling quality
        threshold.
    """
    avg = np.mean([min(a[1], transitions[i + 1, 0])
                   for i, a in enumerate(transitions[:-1])])

    drop = np.tile(False, len(transitions))
    for i, a in enumerate(transitions[:-1]):
        if min(a[1], transitions[i + 1, 0]) < cutoff * avg:
            drop[i] = True

    return drop


def smooth_op(path, order, cutoff):
    """Perform curve fitting on the order parameter free energy
    differences to produce a new estimate of the free energy.

    Args:
        path: The location of the lnpi_op file.
        order: The order of the polynomial used to fit the free
            energy differences.
        cutoff: The fraction of the mean transitions to use as a
            threshold for sampling quality.

    Returns:
        A tuple of numpy arrays containing the subensemble index and
        the new estimate for the free energy of the order parameter
        path.
    """
    index, lnpi = np.loadtxt(path, usecols=(0, 1), unpack=True)
    transitions = np.loadtxt(os.path.join(os.path.dirname(path),
                                          'pacc_op_cr.dat'),
                             usecols=(1, 2))
    drop = find_poorly_sampled(transitions, cutoff)
    diff = np.diff(lnpi)
    p = np.poly1d(np.polyfit(range(len(diff)), diff, order))
    x = index[1:]

    return index, np.append(0.0, np.cumsum(p(x)))


def smooth_tr(path, order, cutoff):
    """Perform curve fitting on the growth expanded ensemble free
    energy differences to produce a new estimate of the free energy.

    Args:
        path: The location of the lnpi_tr file.
        order: The order of the polynomial used to fit the free
            energy differences.
        cutoff: The fraction of the mean transitions to use as a
            threshold for sampling quality.

    Returns:
        A tuple of numpy arrays containing the index, molecule
        number, stage number, and new estimate for the free energy
        of each entry in the expanded ensemble growth path.
    """
    index, sub, mol, stage, lnpi = np.loadtxt(path, usecols=(0, 1, 2, 3, 4),
                                              unpack=True)
    transitions = np.loadtxt(os.path.join(os.path.dirname(path),
                                          'pacc_tr_cr.dat'),
                             usecols=(4, 5))
    drop = find_poorly_sampled(transitions, cutoff)
    diff = np.zeros(len(lnpi))
    fit = np.zeros(len(lnpi))
    new_lnpi = np.zeros(len(lnpi))
    for m in np.unique(mol):
        curr_mol = (mol == m)
        mol_subs = np.unique(sub[curr_mol])
        mol_stages = np.unique(stage[curr_mol])[:-1]
        for s in mol_subs:
            curr_sub = curr_mol & (sub == s)
            max_stage = np.amax(stage[curr_sub])
            diff[curr_sub & (stage < max_stage)] = np.diff(lnpi[curr_sub])

        for i in mol_stages:
            curr_stage = curr_mol & (stage == i)
            y = diff[curr_stage & ~drop]
            p = np.poly1d(np.polyfit(range(len(y)), y, order))
            fit[curr_stage] = p(range(len(lnpi[curr_stage])))

        for s in mol_subs:
            curr_sub = (sub == s)
            for i in reversed(mol_stages):
                curr_stage = curr_mol & curr_sub & (stage == i)
                next_stage = curr_mol & curr_sub & (stage == i + 1)
                new_lnpi[curr_stage] = new_lnpi[next_stage] - fit[curr_stage]

    return index, sub, mol, stage, new_lnpi
