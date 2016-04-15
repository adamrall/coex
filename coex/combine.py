"""Combine the output of several simulations."""

from __future__ import division
import glob
import os.path

import numpy as np

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
    hists = read_histograms_from_runs(path, runs, 'ehist.dat')

    return combine_histograms(hists)


def combine_all_nhists(path, runs):
    hist_files = [os.path.basename(f)
                  for f in glob.glob(os.path.join(runs[0],'nhist_*.dat'))]

    return [combine_histograms(read_histograms_from_runs(path, runs, hf)
            for hf in hist_files]


def combine_vhist(path, runs, uses_log_volume=False):
    hists = read_volume_histograms_from_runs(path, runs, uses_log_volume)

    return combine_histograms(hists)


def combine_hits_op(path, runs):
    index = np.loadtxt(os.path.join(path, runs[0], 'hits_op.dat'),
                       usecols=(0, ))

    return index, sum([np.loadtxt(os.path.join(path, r, 'hits_op.dat'),
                                  usecols=(1, )) for r in runs])


def combine_hits_tr(path, runs):
    index, sub, mol, stage = np.loadtxt(os.path.join(path, runs[0],
                                                     'hits_tr.dat'),
                                        usecols=(0, 1, 2, 3))
    sum_hits = sum([np.loadtxt(os.path.join(path, r, 'hits_tr.dat'),
                               usecols=(4, )) for r in runs])

    return index, sub, mol, stage, sum_hits


def combine_prop(path, runs, file_name):

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

    hits = [np.loadtxt(os.path.join(path, r, hits_file, usecols=cols)
            for r in sorted(runs)]
    index = np.loadtxt(os.path.join(path, runs[0], file_name), usecols=(0, ))
    weighted_sums = np.sum([read_properties(r) * hits[i]
                            for i, r in enumerate(sorted(runs))], axis=0)
    hits_sum = sum(hits)
    hits_sum[hits_sum < 1] = 1.0

    return index, np.transpose(weighted_sums / hits_sum)
