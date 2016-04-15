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
