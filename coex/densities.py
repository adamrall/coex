"""Functions dealing with density histograms."""

import os.path

import numpy as np


class DensityHistogram(object):
    def __init__(self, index, distances, histogram, frequencies):
        self.index = index
        self.distances = distances
        self.histogram = histogram
        self.frequencies = frequencies

    @classmethod
    def from_file(cls, path):
        histogram = np.transpose(np.loadtxt(path))
        dirname = os.path.dirname(path)
        index, frequencies = np.loadtxt(os.path.join(dirname, 'pzcnt.dat'))

        return cls(index=index, distances=histogram[1],
                   histogram=histogram[2:], frequencies=frequencies)

    @classmethod
    def from_combination(cls, hists):
        index, distances = hists[0].index, hists[0].distances
        freq_sum = sum([h.frequencies for h in hists])
        weighted_hist = sum([h.frequencies * np.transpose(h.histogram)
                             for h in hists])
        hist = np.transpose(np.nan_to_num(weighted_hist / freq_sum))

        return DensityHistogram(index=index, distances=distances,
                                histogram=hist, frequencies=freq_sum)

    @classmethod
    def from_combined_runs(cls, path, runs):
        return cls.from_combination([cls.from_file(os.path.join(path, r))
                                     for r in runs])

    def write(self, path, write_pzcnt=False):
        dirname = os.path.dirname(path)
        if write_pzcnt:
            np.savetxt(os.path.join(dirname, 'pzcnt.dat'),
                       np.column_stack(self.index, self.frequencies))

        with open(path, 'w') as f:
            for i, col in enumerate(np.transpose(self.histogram)):
                print(i, self.distances(i), col, file=f)
