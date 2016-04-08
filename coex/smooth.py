"""Functions for smoothing lnpi_op and lnpi_tr."""

import os.path

import numpy as np


def find_poorly_sampled_subensembles(transitions, cutoff):
    """Determine which subensembles are not adequately sampled.

    For each subensemble, we take the minimum of the number of
    forward and backward transitions. If this number is less than the
    average over all subensembles times some cutoff fraction, then we
    add it to the list of poorly sampled subensembles.

    Args:
        transitions: A 2D numpy array with the count of forward and
            backward transitions for each subensemble.
        cutoff: The fraction of the mean to use as a threshold for
            sampling quality.

    Returns:
        A list of subensembles which don't meet the sampling quality
        threshold.
    """
    avg = np.mean(np.amin(transitions, axis=1))

    # Skip the first subensemble.
    return [i for i, a in enumerate(transitions[1:])
            if np.amin(a) < cutoff * avg]


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
    drop = find_poorly_sampled_subensembles(transitions, cutoff)
    diff = np.delete(np.diff(lnpi), drop)
    diff_idx = np.delete(index, drop)
    p = np.poly1d(np.polyfit(diff_idx, diff, order))
    x = index[:-1]
    
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
    transitions = np.loadtxt(os.path.join(os.path.dirname(path)),
                                          'pacc_tr_cr.dat'),
                             usecols=(4, 5))
    for i in np.unique(sub):
        current_sub = np.where(sub == i)
        lnpi_tr[current_sub] -= lnpi_tr[current_sub][-1]

    new_lnpi = np.zeros(len(lnpi))
    for m in np.unique(mol):
        current_mol = np.where(mol == m)
        lnpi_mol = lnpi[current_mol]
        trans_mol = transitions[current_mol]
        stage_mol = stage[current_mol]
        for s in np.unique(stage)[:-1]:
            current_stage = np.where(stage_mol == s)
            drop = find_poorly_sampled_subensembles(trans_mol[current_stage],
                                                    cutoff)
            diff = np.delete(np.diff(lnpi_mol[current_stage]), drop)
            p = np.poly1d(np.polyfit(range(len(diff)), diff, order))
            x = np.unique(sub([:-1]))
            lnpi_mol[current_stage] = (lnpi_mol[current_stage][0] +
                                       np.append(0.0, np.cumsum(p(x))))

        new_lnpi[current_mol] = lnpi_mol

    return index, sub, mol, stage, new_lnpi


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
