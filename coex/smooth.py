"""Functions for smoothing lnpi_op and lnpi_tr."""

import os.path

import numpy as np
import pandas as pd


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
    op = pd.read_table(path, sep='\s+', usecols=(0, 1), header=None,
                       names=('sub', 'lnpi'))
    transitions = np.loadtxt(os.path.join(os.path.dirname(path),
                                          'pacc_op_cr.dat'),
                             usecols=(1, 2))
    op['drop'] = find_poorly_sampled(transitions, cutoff)
    op['diff'] = 0.0
    op.loc[1:, 'diff'] = np.diff(op['lnpi'])
    p = np.poly1d(np.polyfit(op['sub'], op['diff'], order))
    op['new_lnpi'] = np.cumsum(p(op['sub']))
    op['new_lnpi'] -= op.loc[0, 'new_lnpi']

    return op


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
    tr = pd.read_table(path, sep='\s+', usecols=(0, 1, 2, 3, 4), header=None,
                       names=('number', 'sub', 'mol', 'stage', 'lnpi'))
    transitions = np.loadtxt(os.path.join(os.path.dirname(path),
                                          'pacc_tr_cr.dat'),
                             usecols=(4, 5))
    tr['drop'] = find_poorly_sampled(transitions, cutoff)
    tr['diff'] = 0.0
    tr['fit'] = 0.0
    tr['new_lnpi'] = 0.0
    for m in tr['mol'].unique():
        mol = tr['mol'] == m
        for s in tr.loc[mol, 'sub'].unique():
            select = mol & (tr['sub'] == s)
            diff = np.diff(tr.loc[select, 'lnpi'])
            max_stage = tr.loc[mol, 'stage'].max()
            tr.loc[select & (tr['stage'] < max_stage), 'diff'] = diff

        for i in tr.loc[mol, 'stage'].unique()[:-1]:
            select = mol & (tr['stage'] == i)
            y = tr.loc[select & (tr['drop'] == False)]['diff']
            p = np.poly1d(np.polyfit(range(len(y)), y, order))
            x = range(len(tr[select]))
            tr.loc[select, 'fit'] = p(x)

        for s in tr.loc[mol, 'sub'].unique():
            for i in reversed(tr.loc[mol, 'stage'].unique()[:-1]):
                select = mol & (tr['sub'] == s) & (tr['stage'] == i)
                next = mol & (tr['sub'] == s) & (tr['stage'] == i + 1)
                tr.loc[select, 'new_lnpi'] = (tr.loc[next, 'new_lnpi'].values -
                                              tr.loc[select, 'fit'])

    return tr


def write_lnpi_op(path, index, lnpi):
    """Write the new estimate for the free energy of the order
    parameter path to a file.

    Args:
        path: The file to write.
        index: The list of order parameter values.
        lnpi: The logarithm of the probability distribution of the
            order parameter.
    """
    with open(path, 'w') as f:
        for i, p in zip(index, lnpi):
            print(int(i), p, file=f)


def write_lnpi_tr(path, index, sub, mol, stage, lnpi):
    """Write the new estimate for the free energy of the order
    parameter path to a file.

    Each row contains the following information: index, order
    parameter value, molecule type, growth stage, and free energy
    (i.e., the logarithm of the probability distribution of the
    growth expanded ensemble path).

    Args:
        path: The file to write.
        index: A list numbering each entry.
        sub: The list of subensembles (order parameter values).
        mol: The list of molecule types.
        stage: The list of stages.
        lnpi: The logarithm of the probability distribution of the
            growth expanded path.
    """
    with open(path, 'w') as f:
        for i, p in enumerate(lnpi):
            print(int(index[i]), int(sub[i]), int(mol[i]), int(stage[i]), p,
                  file=f)
