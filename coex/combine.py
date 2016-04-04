"""Combine the output of several simulations."""

from __future__ import division

import numpy as np


def combine_histograms(hists):
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
