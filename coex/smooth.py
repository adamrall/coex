"""Functions for smoothing lnpi_op and lnpi_tr."""

import os.path

import numpy as np


def find_poorly_sampled(attempts, cutoff):
    """Determine which subensemble/molecule/growth stage combinations
    are not adequately sampled.

    For each combination, we take the minimum of the number of
    forward and backward transitions. If this number is less than the
    average over all combinations times some cutoff fraction, then we
    add it to the list of poorly sampled combinations.

    Args:
        attempts: A 2D numpy array with the count of forward and
            reverse transition attempts for each subensemble.
        cutoff: The fraction of the mean to use as a threshold for
            sampling quality.

    Returns:
        A list of indices which don't meet the sampling quality
        threshold.
    """
    avg = np.mean([min(a[1], attempts[i + 1, 0])
                   for i, a in enumerate(attempts[:-1])])

    drop = np.tile(False, len(attempts))
    for i, a in enumerate(attempts[:-1]):
        if min(a[1], attempts[i + 1, 0]) < cutoff * avg:
            drop[i] = True

    return drop


def smooth_op(op, order, drop=None):
    """Perform curve fitting on the order parameter free energy
    differences to produce a new estimate of the free energy.

    Args:
        op: A dict containing the order parameter values and free
            energy.
        order: The order of the polynomial used to fit the free
            energy differences.
        drop: A boolean numpy array denoting whether to drop each
            subensemble prior to fitting.

    Returns:
        A dict containing the subensemble index and the new estimate
        for the free energy of the order parameter path.
    """
    if drop is None:
        drop = np.tile(False, len(op['lnpi']))

    diff = np.diff(op['lnpi'])
    y = diff[~drop]
    p = np.poly1d(np.polyfit(range(len(y)), y, order))

    return {'index': op['index'],
            'lnpi': np.append(0.0, np.cumsum(p(op['index'][1:])))}


def smooth_tr(tr, order, drop=None):
    """Perform curve fitting on the growth expanded ensemble free
    energy differences to produce a new estimate of the free energy.

    Args:
        tr: A dict containing the growth expanded ensemble data; see
            coex.read.read_lnpi_tr().
        order: The order of the polynomial used to fit the free
            energy differences.
        drop: A boolean numpy array denoting whether to drop each
            entry prior to fitting.

    Returns:
        A dict containing the index, molecule number, stage number,
        and new estimate for the free energy of each entry in the
        expanded ensemble growth path.
    """
    size = len(tr['lnpi'])
    mol, sub, stage = tr['mol'], tr['sub'], tr['stage']
    diff, fit, new_lnpi = np.zeros(size), np.zeros(size), np.zeros(size)
    if drop is None:
        drop = np.tile(False, size)

    for m in np.unique(mol):
        curr_mol = (mol == m)
        mol_subs = np.unique(sub[curr_mol])
        mol_stages = np.unique(stage[curr_mol])[:-1]
        for s in mol_subs:
            curr_sub = curr_mol & (sub == s)
            not_max = stage < np.amax(stage[curr_sub])
            diff[curr_sub & not_max] = np.diff(tr['lnpi'][curr_sub])

        for i in mol_stages:
            curr_stage = curr_mol & (stage == i)
            y = diff[curr_stage & ~drop]
            p = np.poly1d(np.polyfit(range(len(y)), y, order))
            fit[curr_stage] = p(range(len(curr_stage)))

        for s in mol_subs:
            curr_sub = (sub == s)
            for i in reversed(mol_stages):
                curr_stage = curr_mol & curr_sub & (stage == i)
                next_stage = curr_mol & curr_sub & (stage == i + 1)
                new_lnpi[curr_stage] = new_lnpi[next_stage] - fit[curr_stage]

    out = dict.copy(tr)
    out['lnpi'] = new_lnpi

    return out
