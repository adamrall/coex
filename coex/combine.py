"""Combine the output of several simulations."""

import glob
import os.path

import numpy as np

from coex.read import read_pacc_op, read_pacc_tr
from coex.states import read_histograms_from_runs
from coex.states import read_volume_histograms_from_runs


def combine_histograms(hists):
    """Combine a set of visited states histograms.

    Args:
        hists: A list of histograms.

    Returns:
        A histogram with the combined data.
    """
    first_dist = hists[0][0]
    step = first_dist['bins'][1] - first_dist['bins'][0]
    subensembles = len(first_dist['bins'])

    def combine_subensemble(i):
        min_bin = min([h[i]['bins'][0] for h in hists])
        max_bin = max([h[i]['bins'][-1] for h in hists])
        num = int((max_bin - min_bin) / step) + 1
        bins = np.linspace(min_bin, max_bin, num)
        counts = np.zeros(num)
        for h in hists:
            shift = int((h[i]['bins'][0] - min_bin) / step)
            counts[shift:] = h[i]['counts']

        return {'bins': bins, 'counts': counts}

    return [combine_subensemble(i) for i in range(subensembles)]


def combine_ehist(path, runs):
    """Combine a set of energy visited states histograms.

    Args:
        path: The base path containing the data to combine.
        runs: The list of runs to combine.

    Returns:
        A combined energy histogram.
    """
    return combine_histograms(read_histograms_from_runs(path, runs,
                                                        'ehist.dat'))


def combine_all_nhists(path, runs):
    """Combine all molecule number visited states histograms from a
    set of runs.

    Args:
       path: The base path containing the data to combine.
       runs: The list of runs to combine.

    Returns:
        A list of combined histograms, with one entry for each
        species.
    """
    hist_files = [os.path.basename(f)
                  for f in glob.glob(os.path.join(runs[0],'nhist_*.dat'))]

    return [combine_histograms(read_histograms_from_runs(path, runs, hf))
            for hf in hist_files]


def combine_vhist(path, runs, uses_log_volume=False):
    """Combine a set of energy visited states histograms.

    Args:
        path: The base path containing the data to combine.
        runs: The list of runs to combine.
        uses_log_volume: A bool denoting whether the bins in the
            histogram use V or ln(V).

    Returns:
        A combined volume histogram.
    """
    hists = read_volume_histograms_from_runs(path, runs, uses_log_volume)

    return combine_histograms(hists)


def combine_hits_op(path, runs):
    """Combine a set of order parameter hits files.

    Args:
        path: The base path containing the data to combine.
        runs: The list of runs to combine.

    Returns:
        A dict containing the order parameter values and the combined
        hits for each value.
    """
    return {'index': np.loadtxt(os.path.join(path, runs[0], 'hits_op.dat'),
                                usecols=(0, ), dtype='int'),
            'hits': sum([np.loadtxt(os.path.join(path, r, 'hits_op.dat'),
                                    usecols=(1, ), dtype='int')
                         for r in runs])}


def combine_hits_tr(path, runs):
    """Combine a set of growth expanded ensemble hits files.

    Args:
        path: The base path containing the data to combine.
        runs: The list of runs to combine.

    Returns:
        A dict containing the first five columns of the combined
        hits_tr.dat file, i.e., the index, order parameter value,
        species ID, growth stage, and combined number of hits.
    """
    index, sub, mol, stage = np.loadtxt(os.path.join(path, runs[0],
                                                     'hits_tr.dat'),
                                        usecols=(0, 1, 2, 3), dtype='int')

    return {'index': index, 'sub': sub, 'mol': mol, 'stage': stage,
            'hits': sum([np.loadtxt(os.path.join(path, r, 'hits_tr.dat'),
                                    usecols=(4, ), dtype='int')
                         for r in runs])}


def combine_prop(path, runs, file_name):
    """Combine a set of property files.

    Args:
        path: The base path containing the data to combine.
        runs: The list of runs to combine.
        file_name: The file to combine: 'prop_op.dat', 'prop_tr.dat',
            or 'prop_ex.dat'.

    Returns:
        A dict containing the index column and a numpy array with the
        remaining columns of the combined property files.
    """

    def read_properties(run):
        return np.transpose(np.loadtxt(os.path.join(path, run, file_name)))[1:]

    if 'op' in file_name:
        hits_file = 'hits_op.dat'
        cols = (1, )
    elif 'tr' in file_name:
        hits_file = 'hits_tr.dat'
        cols = (4, )
    elif 'ex' in file_name:
        hits_file = 'hits_ex.dat'
        cols = (5, )

    hits = [np.loadtxt(os.path.join(path, r, hits_file), dtype='int',
                       usecols=cols) for r in sorted(runs)]
    index = np.loadtxt(os.path.join(path, runs[0], file_name), dtype='int',
                       usecols=(0, ))
    weighted_sums = np.sum([read_properties(r) * hits[i]
                            for i, r in enumerate(sorted(runs))], axis=0)
    hits_sum = sum(hits)
    hits_sum[hits_sum < 1] = 1.0

    return {'index': index, 'prop': np.transpose(weighted_sums / hits_sum)}


def combine_pzcnt(path, runs):
    """Combine a set of density histogram count files.

    Args:
        path: The base path containing the data to combine.
        runs: The list of runs to combine.

    Returns:
        A dict containing the index and combined density histogram
        counts for each order parameter value.
    """
    return {'index': np.loadtxt(os.path.join(path, runs[0], 'pzcnt.dat'),
                                usecols=(0, ), dtype='int'),
            'counts': sum([np.loadtxt(os.path.join(path, r, 'pzcnt.dat'),
                                                   usecols=(1, ))
                           for r in runs])}


def combine_all_pzhists(path, runs):
    """Combine all denisty histograms for a set of runs.

    Args:
        path: The base path containing the data to combine.
        runs: The list of runs to combine.

    Returns:
        A dict containing the density histogram index, z distance
        bins, and the combined histograms, indexed by file name.
    """
    index, z_bins = np.loadtxt(os.path.join(path, runs[0], 'pzhist_01_01.dat'),
                               usecols=(0, 1), unpack=True)
    cnts = [np.loadtxt(os.path.join(path, r, 'pzcnt.dat'), usecols=(1, ))]
    cnts_sum = susm(cnts)
    cnts_sum[cnts_sum < 1] = 1.0

    def read_pzhist(run, hist_file):
        return np.loadtxt(os.path.join(path, run, hist_file))[:, 2:]

    def combine_pzhist(hist_file):
        weighted_sums = np.sum([read_pzhist(r, hist_file) * cnts[i]
                                for i, r in enumerate(sorted(runs))], axis=0)
        return weighted_sums / cnts_sum

    hist_files = sorted([os.path.basename(f)
                         for f in glob.glob(os.path.join(path, runs[0],
                                                         'pzhist_*.dat'))])
    out = {hf: combine_pzhist(hf) for hf in hist_files}
    out['index'] = index.astype('int')
    out['z_bins'] = z_bins

    return out


def combine_pacc_op(path, runs):
    """Combine a set of order parameter transition matrix files.

    Args:
        path: The base path containing the data to combine.
        runs: The list of runs to combine.

    Returns:
        A dict containing the index column, a numpy array with the
        transition attempts, and a numpy array with the acceptance
        probabilities.
    """
    paccs = [read_pacc_op(os.path.join(path, r, 'pacc_op_cr.dat'))
             for r in runs]
    out = {'index': paccs[0]['index']}
    out['attempts'] = np.sum([p['attempts'] for p in paccs], axis=0)
    out['prob'] = (np.sum([p['attempts'] * p['prob'] for p in paccs], axis=0) /
                   out['attempts'])
    out['prob'] = np.nan_to_num(out['prob'])

    return out


def combine_pacc_tr(path, runs):
    """Combine a set of growth expanded ensemble transition matrix
    files.

    Args:
        path: The base path containing the data to combine.
        runs: The list of runs to combine.

    Returns:
        A dict containing the index, order parameter value, species
        ID, and growth stage columns, a numpy array with the
        transition attempts, and a numpy array with the acceptance
        probabilities.
    """
    paccs = [read_pacc_tr(os.path.join(path, r, 'pacc_tr_cr.dat'))
             for r in runs]
    out = {'index': paccs[0]['index'], 'mol': paccs[0]['mol'],
           'sub': paccs[0]['sub'], 'stage': paccs[0]['stage']}
    out['attempts'] = np.sum([p['attempts'] for p in paccs], axis=0)
    out['prob'] = (np.sum([p['attempts'] * p['prob'] for p in paccs], axis=0) /
                   out['attempts'])
    out['prob'] = np.nan_to_num(out['prob'])

    return out


def compute_lnpi_op(pacc, guess=None, min_attempts=1):
    """Compute the order parameter free energy using the transition
    matrix.

    Args:
        pacc: A dict with the transition matrix data.
        guess: An initial guess for the free energy.
        min_attempts: The threshold for considering a transition
            adequately sampled.

    Returns:
        A dict with the order parameter values and the computed order
        parameter free energy.
    """
    lnpi = np.zeros(len(pacc['index']))
    if guess is None:
        guess = np.copy(lnpi)

    att, prob = pacc['attempts'], pacc['prob']
    for i, dc in enumerate(np.diff(guess)):
        lnpi[i + 1] = lnpi[i] + dc
        if (att[i, 1] > min_attempts and att[i + 1, 0] > min_attempts and
                prob[i, 1] > 0.0 and prob[i + 1, 0] > 0.0):
            lnpi[i + 1] += np.log(prob[i, 1] / prob[i + 1, 0])

    return {'index': pacc['index'], 'lnpi': lnpi}


def compute_lnpi_tr(pacc, guess=None, min_attempts=1):
    """Compute the growth expanded ensemble free energy using the
    transition matrix.

    Args:
        pacc: A dict with the transition matrix data.
        guess: An initial guess for the free energy.
        min_attempts: The threshold for considering a transition
            adequately sampled.

    Returns:
        A dict with the index, subensemble number, molecule number,
        stage number, and the computed growth expanded ensemble free
        energy.
    """
    lnpi = np.zeros(len(pacc['index']))
    if guess is None:
        guess = np.copy(lnpi)

    att, prob = pacc['attempts'], pacc['prob']
    mol, sub, stage = pacc['mol'], pacc['sub'], pacc['stage']
    for m in np.unique(mol):
        for s in np.unique(sub):
            sel = (mol == m) & (sub == s)
            if len(stage[sel]) == 1:
                continue

            for g in stage[sel][-2::-1]:
                curr = sel & (stage == g)
                next = sel & (stage == g + 1)
                lnpi[curr] = lnpi[next] + guess[curr] - guess[next]
                if (att[curr, 1] > min_attempts and
                        att[next, 0] > min_attempts and prob[curr, 1] > 0.0 and
                        prob[next, 0] > 0.0):
                    lnpi[curr] -= np.log(prob[curr, 1] / prob[next, 0])

    return {'index': pacc['index'], 'mol': pacc['mol'], 'sub': pacc['sub'],
            'stage': pacc['stage'], 'lnpi': lnpi}
