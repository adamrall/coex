"""Functions dealing with density histograms."""

import glob
import os.path

import numpy as np


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
                                      usecols=(1, )) for r in runs])}


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
    cnts = [np.loadtxt(os.path.join(path, r, 'pzcnt.dat'), usecols=(1, ))
            for r in runs]
    cnts_sum = sum(cnts)
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
