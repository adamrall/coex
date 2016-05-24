"""Objects and functions for working with property distributions.

Internally, we often deal with the logarithm of the probability
distribution along a path of interest instead of the free energy
differences, which differ only by a minus sign.

In gchybrid, we refer to the logarithm of the order parameter
distribution as lnpi_op.dat, the logarithm of the growth expanded
ensemble distribution as lnpi_tr.dat, the logarithm of the exchange
path distribution as lnpi_ex.dat, and the logarithm of the regrowth
path distribution as lnpi_rg.dat.  Similarly, the number of samples
and other properties along each path are contained in the files
hits_*.dat and prop_*.dat, respectively, where the * matches the
appropriate two-letter suffix.
"""

import copy
import os.path

import numpy as np


class TransitionMatrix(object):
    def __init__(self, index, forward_attempts, reverse_attempts,
                 forward_probabilities, reverse_probabilities):
        self.index = index
        self.forward_attempts = forward_attempts
        self.reverse_attempts = reverse_attempts
        self.forward_probabilities = forward_probabilities
        self.reverse_probabilities = reverse_probabilities

    def __len__(self):
        return len(self.forward_attempts)

    def get_poorly_sampled_attempts(self, cutoff):
        """Determine which subensemble/molecule/growth stage
        combinations are not adequately sampled.

        For each combination, we take the minimum of the number of
        forward and backward transition attempts. If this number is
        less than the average over all combinations times some cutoff
        fraction, then we add it to the list of poorly sampled
        combinations.

        Args:
            cutoff: The fraction of the mean to use as a threshold for
                sampling quality.

        Returns:
            A boolean numpy array, where True denotes indices which
            don't meet the sampling quality threshold.
        """
        fw_att, rev_att = self.forward_attempts, self.reverse_attempts
        avg = np.mean([min(a, rev_att[i + 1]) for i, a in enumerate(fw_att)])

        drop = np.tile(False, len(fw_att))
        drop[-1] = True
        for i, a in enumerate(fw_att[:-1]):
            if min(a, rev_att[i + 1]) < cutoff * avg:
                drop[i] = True

        return drop


def _combine_transition_matrices(matrices):
    fw_att = sum([m.forward_attempts for m in matrices])
    rev_att = sum([m.reverse_attempts for m in matrices])
    fw_prob = sum([m.forward_attempts * m.forward_probabilities
                   for m in matrices]) / fw_att
    rev_prob = sum([m.reverse_attempts * m.reverse_probabilities
                    for m in matrices]) / rev_att

    return fw_att, rev_att, fw_prob, rev_prob


def _read_growth_expanded_index(path):
    num, sub, mol, stg = np.loadtxt(path, usecols=(0, 1, 2, 3),
                                    dtype='int', unpack=True)

    return {'numbers': num, 'subensembles': sub, 'molecules': mol,
            'stages': stg}


def compute_growth_expanded_distribution(matrix, guess=None, min_attempts=1):
    """Compute the logarithm of the growth expanded ensemble
    probability distribution using the transition matrix.

    Args:
        matrix: A TransitionMatrix object.
        guess: An initial guess for the logarithm of the probability
            distribution.
        min_attempts: The threshold for considering a transition
            adequately sampled.

    Returns:
        A GrowthExpandedDistribution object with the computed
        logarithm of the probability distribution
    """
    dist = np.zeros(len(matrix.index))
    if guess is None:
        guess = np.copy(dist)

    fw_att, rev_att = matrix.forward_attempts, matrix.reverse_attempts
    fw_prob = matrix.forward_probabilities
    rev_prob = matrix.reverse_probabilities
    mi = matrix.index
    mol, sub, stages = mi['molecules'], mi['subensembles'], mi['stages']
    for m in np.unique(mol):
        for s in np.unique(sub):
            sel = (mol == m) & (sub == s)
            if len(stages[sel]) == 1:
                continue

            for g in stages[sel][-2::-1]:
                cs = sel & (stages == g)
                ns = sel & (stages == g + 1)
                dist[cs] = dist[ns] + guess[cs] - guess[ns]
                if (fw_att[cs] > min_attempts and
                        rev_att[ns] > min_attempts and
                        fw_prob[cs] > 0.0 and rev_prob[ns] > 0.0):
                    dist[cs] -= np.log(fw_prob[cs] / rev_prob[ns])

    return GrowthExpandedDistribution(index=mi, log_probabilities=dist)


def combine_growth_expanded_transition_matrices(path, runs):
    """Combine a set of growth expanded ensemble transition matrix
    files.

    Args:
        path: The base path containing the data to combine.
        runs: The list of runs to combine.

    Returns:
        A GrowthExpandedTransitionMatrix object with the combined
        data.
    """
    matrix_file = 'pacc_tr_cr.dat'
    matrices = [read_growth_expanded_transition_matrix(
        os.path.join(path, r, matrix_file)) for r in runs]
    index = _read_growth_expanded_index(os.path.join(path, runs[0],
                                                     matrix_file))
    fw_att, rev_att, fw_prob, rev_prob = _combine_transition_matrices(matrices)
    fw_prob, rev_prob = np.nan_to_num(fw_prob), np.nan_to_num(rev_prob)

    return TransitionMatrix(
        index, forward_attempts=fw_att, reverse_attempts=rev_att,
        forward_probabilities=fw_prob, reverse_probabilities=rev_prob)


def read_growth_expanded_transition_matrix(path):
    """Read the growth expanded ensemble transition matrix from a
    pacc_tr_cr.dat or pacc_tr_ag.dat file.

    Args:
        path: The location of the file.

    Returns:
        A GrowthExpandedTransitionMatrix object.
    """
    index = _read_growth_expanded_index(path)
    fw_att, rev_att, fw_prob, rev_prob = np.loadtxt(
        path, usecols=(4, 5, 6, 7), unpack=True)

    return TransitionMatrix(
        index, forward_attempts=fw_att.astype('int'),
        reverse_attempts=rev_att.astype('int'), forward_probabilities=fw_prob,
        reverse_probabilities=rev_prob)


def compute_order_parameter_distribution(matrix, guess=None, min_attempts=1):
    """Compute the logarithm of the order parameter probability
    distribution, i.e., the negative of the free energy, using the
    transition matrix.

    Args:
        matrix: A TransitionMatrix object.
        guess: An initial guess for the log of the probabilities.
        min_attempts: The threshold for considering a transition
            adequately sampled.

    Returns:
        An OrderParameterDistribution object.
    """
    dist = np.zeros(len(matrix))
    if guess is None:
        guess = np.copy(dist)

        for i, dc in enumerate(np.diff(guess)):
            dist[i + 1] = dist[i] + dc
            fw_prob = matrix.forward_probabilities[i]
            rev_prob = matrix.reverse_probabilities[i + 1]
            if (matrix.forward_attempts[i] > min_attempts and
                    matrix.reverse_attempts[i + 1] > min_attempts and
                    fw_prob > 0.0 and rev_prob > 0.0):
                dist[i + 1] += np.log(fw_prob / rev_prob)

    return OrderParameterDistribution(index=matrix.index,
                                      log_probabilities=dist)


def combine_order_parameter_transition_matrices(path, runs):
    """Combine a set of order parameter transition matrix files.

    Args:
        path: The base path containing the data to combine.
        runs: The list of runs to combine.

    Returns:
        An OrderParameterTransitionMatrix object with the combined
        data.
    """
    matrix_file = 'pacc_op_cr.dat'
    matrices = [read_order_parameter_transition_matrix(
        os.path.join(path, r, matrix_file)) for r in runs]
    index = np.loadtxt(os.path.join(path, runs[0], matrix_file))
    fw_att, rev_att, fw_prob, rev_prob = _combine_transition_matrices(matrices)

    return TransitionMatrix(index, forward_attempts=fw_att,
                            reverse_attempts=rev_att,
                            forward_probabilities=fw_prob,
                            reverse_probabilities=rev_prob)


def read_order_parameter_transition_matrix(path):
    """Read the order parameter transition matrix from a
    pacc_op_cr.dat or pacc_op_ag.dat file.

    Args:
        path: The location of the file.

    Returns:
        A TransitionMatrix object.
    """
    raw = np.loadtxt(path, usecols=(0, 1, 2, 3, 4)).transpose()
    index, fw_att, rev_att = [c.astype('int') for c in raw[:3]]
    fw_prob, rev_prob = raw[3:]

    return TransitionMatrix(index, forward_attempts=fw_att,
                            reverse_attempts=rev_att,
                            forward_probabilities=fw_prob,
                            reverse_probabilities=rev_prob)


def _write_growth_expanded_distribution(path, index, properties):
    with open(path, 'w') as f:
        for i, sub, mol, stg, p in zip(
                index['numbers'], index['subensembles'], index['molecules'],
                index['stages'], properties):
            print(i, sub, mol, stg, p, file=f)


def _write_order_parameter_distribution(path, index, properties):
    with open(path, 'w') as f:
        for i, p in zip(index, properties):
            print(i, p, file=f)


class Distribution(object):
    def __init__(self, index, log_probabilities):
        self.index = index
        self.log_probabilities = log_probabilities

    def __len__(self):
        return len(self.log_probabilities)

    def __getitem__(self, i):
        return self.log_probabilities[i]

    def __iter__(self):
        for p in self.log_probabilities:
            yield p


class GrowthExpandedDistribution(Distribution):
    def __init__(self, index, log_probabilities):
        super(GrowthExpandedDistribution, self).__init__(index,
                                                         log_probabilities)

    def smooth(self, order, drop=None):
        """Perform curve fitting on the growth expanded ensemble free
        energy differences to produce a new estimate of the free
        energy.

        Args:
            order: The order of the polynomial used to fit the free
                energy differences.
            drop: A boolean numpy array denoting whether to drop each
                entry prior to fitting.

        Returns:
            A dict containing the index, molecule number, stage
            number, and new estimate for the free energy of each entry
            in the expanded ensemble growth path.
        """
        size = len(self)
        mol, sub, stage = self.molecules, self.subensembles, self.stages
        diff, fit = np.zeros(size), np.zeros(size)
        dist = np.zeros(size)
        if drop is None:
            drop = np.tile(False, size)

        for m in np.unique(mol):
            curr_mol = (mol == m)
            mol_subs = np.unique(sub[curr_mol])
            mol_stages = np.unique(stage[curr_mol])[:-1]
            for s in mol_subs:
                curr_sub = curr_mol & (sub == s)
                not_max = stage < np.amax(stage[curr_sub])
                diff[curr_sub & not_max] = np.diff(self[curr_sub])

            for i in mol_stages:
                curr_stage = curr_mol & (stage == i)
                y = diff[curr_stage & ~drop]
                p = np.poly1d(np.polyfit(range(len(y)), y, order))
                fit[curr_stage] = p(range(len(fit[curr_stage])))

            for s in mol_subs:
                curr_sub = (sub == s)
                for i in reversed(mol_stages):
                    curr_stage = curr_mol & curr_sub & (stage == i)
                    next_stage = curr_mol & curr_sub & (stage == i + 1)
                    dist[curr_stage] = dist[next_stage] - fit[curr_stage]

        smoothed = copy.copy(self)
        smoothed.log_probabilities = dist

        return smoothed

    def write(self, path):
        """Write the new free energy to a file.

        Args:
            path: The file to write.
        """
        _write_growth_expanded_distribution(path, self.index,
                                            self.log_probabilities)


def read_growth_expanded_distribution(path):
    """Read the logarithm of the growth expanded ensemble probability
    distribution from an lnpi_tr.dat file.

    Args:
        path: The location of the file.

    Returns:
        A GrowthExpandedDistribution object.
    """
    index = _read_growth_expanded_index(path)
    logp = np.loadtxt(path, usecols=(4, ))

    return GrowthExpandedDistribution(index=index, log_probabilities=logp)


class OrderParameterDistribution(Distribution):
    def __init__(self, index, log_probabilities):
        super(OrderParameterDistribution, self).__init__(index,
                                                         log_probabilities)

    def smooth(self, order, drop=None):
        """Perform curve fitting on the free energy differences to
        produce a new estimate of the free energy.

        Args:
            order: The order of the polynomial used to fit the free
                energy differences.
            drop: A boolean numpy array denoting whether to drop each
                subensemble prior to fitting.

        Returns:
            A dict containing the subensemble index and the new
            estimate for the free energy of the order parameter path.
        """
        if drop is None:
            drop = np.tile(False, len(self) - 1)

        x = self.index[~drop]
        y = np.diff(self[~drop])
        p = np.poly1d(np.polyfit(x, y, order))
        smoothed = np.append(0.0, np.cumsum(p(self.index[1:])))

        return OrderParameterDistribution(index=self.index,
                                          log_probabilities=smoothed)

    def write(self, path):
        """Write the new free energy to a file.

        Args:
            path: The file to write.
        """
        _write_order_parameter_distribution(path, self.index,
                                            self.log_probabilities)


def read_order_parameter_distribution(path):
    """Read the logarithm of the order parameter probability
    distribution from an lnpi_op.dat file.

    Args:
        path: The location of the file.

    Returns:
        A dict with the order parameter values and free energy.
    """
    index, log_probabilities = np.loadtxt(path, usecols=(0, 1), unpack=True)

    return OrderParameterDistribution(index=index.astype('int'),
                                      log_probabilities=log_probabilities)


class GrowthExpandedSamples(object):
    def __init__(self, index, samples):
        self.index = index
        self.samples = samples

    def write(self, path):
        _write_growth_expanded_distribution(path, self.index, self.samples)


def combine_growth_expanded_samples(path, runs):
    """Combine a set of growth expanded ensemble sample count
    (hits_tr.dat) files.

    Args:
        path: The base path containing the data to combine.
        runs: The list of runs to combine.

    Returns:
        A GrowthExpandedSamples object with the combined data.
    """
    sample_file = 'hits_tr.dat'
    index = _read_growth_expanded_index(os.path.join(path, runs[0],
                                                     sample_file))
    samples = sum([np.loadtxt(os.path.join(path, r, sample_file),
                              usecols=(4, ), dtype='int') for r in runs])

    return GrowthExpandedSamples(index=index, samples=samples)


def read_growth_expanded_samples(path):
    index = _read_growth_expanded_index(path)
    samples = np.loadtxt(path, usecols=(4, ), dtype='int', unpack=True)

    return GrowthExpandedSamples(index=index, samples=samples)


class OrderParameterSamples(object):
    def __init__(self, index, samples):
        self.index = index
        self.samples = samples

    def write(self, path):
        _write_order_parameter_distribution(path, self.index, self.samples)


def combine_order_parameter_samples(path, runs):
    """Combine a set of order parameter sample count (hits_op.dat)
    files.

    Args:
        path: The base path containing the data to combine.
        runs: The list of runs to combine.

    Returns:
        An OrderParameterSamples object with the combined data.
    """
    sample_file = 'hits_op.dat'
    index = np.loadtxt(os.path.join(path, runs[0], sample_file),
                       usecols=(0, ), dtype='int'),
    samples = sum([np.loadtxt(os.path.join(path, r, sample_file),
                              usecols=(1, ), dtype='int') for r in runs])

    return OrderParameterSamples(index=index, samples=samples)


def read_order_parameter_samples(path):
    index, samples = np.loadtxt(path, usecols=(0, 1), dtype='int', unpack=True)

    return OrderParameterSamples(index=index, samples=samples)


def _read_properties(path):
    base = os.path.basename(path)
    if 'op' in base:
        return np.transpose(np.loadtxt(path))[1:]
    elif 'tr' in base:
        return np.transpose(np.loadtxt(path))[4:]
    else:
        raise NotImplementedError


def _combine_properties(path, runs, sample_file):
    run_samples = [read_order_parameter_samples(os.path.join(path, r,
                                                             sample_file))
                   for r in runs]
    weighted_sums = np.sum([_read_properties(r) *
                            run_samples[i].samples
                            for i, r in enumerate(sorted(runs))], axis=0)
    samples_sum = sum([s.samples for s in run_samples])
    samples_sum[samples_sum < 1] = 1.0

    return np.transpose(weighted_sums / samples_sum)


def combine_growth_expanded_properties(path, runs):
    prop_file = 'prop_tr.dat'
    index = _read_growth_expanded_index(os.path.join(path, runs[0],
                                                     prop_file))
    prop = _combine_properties(path, runs, sample_file='hits_tr.dat')
    pass


def read_growth_expanded_properties(path):
    pass
