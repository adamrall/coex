"""Objects and functions for working with probability distributions
and related properties.

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
    matrix_file = None

    def __init__(self, index, forward_attempts, reverse_attempts,
                 forward_probabilities, reverse_probabilities):
        self.index = index
        self.forward_attempts = forward_attempts
        self.reverse_attempts = reverse_attempts
        self.forward_probabilities = forward_probabilities
        self.reverse_probabilities = reverse_probabilities

    def __len__(self):
        return len(self.forward_attempts)

    @classmethod
    def from_combination(cls, matrices):
        index = matrices[0].index
        fw_att = sum([m.forward_attempts for m in matrices])
        rev_att = sum([m.reverse_attempts for m in matrices])
        fw_prob = sum([m.forward_attempts * m.forward_probabilities
                       for m in matrices]) / fw_att
        rev_prob = sum([m.reverse_attempts * m.reverse_probabilities
                        for m in matrices]) / rev_att
        fw_prob, rev_prob = np.nan_to_num(fw_prob), np.nan_to_num(rev_prob)

        return cls(
            index=index, forward_attempts=fw_att, reverse_attempts=rev_att,
            forward_probabilities=fw_prob, reverse_probabilities=rev_prob)

    @classmethod
    def from_combined_runs(cls, path, runs):
        """Combine a set of transition matrix files.

        Args:
            path: The base path containing the data to combine.
            runs: The list of runs to combine.

        Returns:
            A TransitionMatrix object with the combined data.
        """
        return cls.from_combination(
            [cls.from_file(os.path.join(path, r, cls.matrix_file))
             for r in runs])

    @classmethod
    def from_file(cls, path):
        raise NotImplementedError

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
        fw, rev = self.forward_attempts, self.reverse_attempts
        avg = np.mean([min(a, rev[i + 1]) for i, a in enumerate(fw[:-1])])

        drop = np.tile(False, len(fw))
        drop[-1] = True
        for i, a in enumerate(fw[:-1]):
            if min(a, rev[i + 1]) < cutoff * avg:
                drop[i] = True

        return drop


def _read_growth_expanded_index(path):
    num, sub, mol, stg = np.loadtxt(path, usecols=(0, 1, 2, 3),
                                    dtype='int', unpack=True)

    return {'numbers': num, 'subensembles': sub, 'molecules': mol,
            'stages': stg}


class GrowthExpandedTransitionMatrix(TransitionMatrix):
    matrix_file = 'pacc_tr_cr.dat'

    def __init__(self, index, forward_attempts, reverse_attempts,
                 forward_probabilities, reverse_probabilities):
        super(GrowthExpandedTransitionMatrix, self).__init__(
            index, forward_attempts, reverse_attempts, forward_probabilities,
            reverse_probabilities)

    def calculate_distribution(self, guess=None, min_attempts=1):
        """Compute the logarithm of the growth expanded ensemble
        probability distribution using the transition matrix.

        Args:
            guess: An initial guess for the logarithm of the
                probability distribution.
            min_attempts: The threshold for considering a transition
                adequately sampled.

        Returns:
            A GrowthExpandedDistribution object with the computed
            logarithm of the probability distribution
        """
        dist = np.zeros(len(self))
        if guess is None:
            guess = np.copy(dist)

        fw_att, rev_att = self.forward_attempts, self.reverse_attempts
        fw_prob = self.forward_probabilities
        rev_prob = self.reverse_probabilities
        ind = self.index
        mol, sub, stages = ind['molecules'], ind['subensembles'], ind['stages']
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

        return GrowthExpandedDistribution(index=ind, log_probabilities=dist)

    @classmethod
    def from_file(cls, path):
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

        return cls(
            index, forward_attempts=fw_att.astype('int'),
            reverse_attempts=rev_att.astype('int'),
            forward_probabilities=fw_prob, reverse_probabilities=rev_prob)

    def write(self, path):
        ind = self.index
        fmt = 6 * ['%.8d'] + 2 * ['%10.5g']
        arr = np.append(ind['number'], ind['subensembles'], ind['molecules'],
                        ind['stages'], self.forward_attempts,
                        self.reverse_attempts, self.forward_probabilities,
                        self.reverse_probabilities)
        np.savetxt(path, np.transpose(arr), fmt=fmt, delimiter='  ')


class OrderParameterTransitionMatrix(TransitionMatrix):
    matrix_file = 'pacc_op_cr.dat'

    def __init__(self, index, forward_attempts, reverse_attempts,
                 forward_probabilities, reverse_probabilities):
        super(OrderParameterTransitionMatrix, self).__init__(
            index, forward_attempts, reverse_attempts, forward_probabilities,
            reverse_probabilities)

    def calculate_distribution(self, guess=None, min_attempts=1):
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
        dist = np.zeros(len(self))
        if guess is None:
            guess = np.copy(dist)

        for i, dc in enumerate(np.diff(guess)):
            dist[i + 1] = dist[i] + dc
            fw_prob = self.forward_probabilities[i]
            rev_prob = self.reverse_probabilities[i + 1]
            if (self.forward_attempts[i] > min_attempts and
                    self.reverse_attempts[i + 1] > min_attempts and
                    fw_prob > 0.0 and rev_prob > 0.0):
                dist[i + 1] += np.log(fw_prob / rev_prob)

        return OrderParameterDistribution(index=self.index,
                                          log_probabilities=dist)

    @classmethod
    def from_file(cls, path):
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

        return cls(
            index, forward_attempts=fw_att, reverse_attempts=rev_att,
            forward_probabilities=fw_prob, reverse_probabilities=rev_prob)

    def write(self, path):
        fmt = 3 * ['%.8d'] + 2 * ['%10.5g']
        arr = np.append(
            self.index, self.forward_attempts, self.reverse_attempts,
            self.forward_probabilities, self.reverse_probabilities)
        np.savetxt(path, np.transpose(arr), fmt=fmt, delimiter='  ')


def _write_growth_expanded_distribution(path, index, probabilities):
    fmt = 4 * ['%.8d'] + ['%10.5g']
    arr = np.append(index['numbers'], index['subensembles'],
                    index['molecules'], index['stages'], probabilities)
    np.savetxt(path, np.transpose(arr), fmt=fmt, delimiter='  ')


def _write_order_parameter_distribution(path, index, probabilities):
    np.savetxt(path, np.transpose(np.append(index, probabilities)),
               fmt=['%.8d', '%10.5g'], delimiter='  ')


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

    @classmethod
    def from_file(cls, path):
        """Read the logarithm of the growth expanded ensemble probability
        distribution from an lnpi_tr.dat file.

        Args:
            path: The location of the file.

        Returns:
            A GrowthExpandedDistribution object.
        """
        index = _read_growth_expanded_index(path)
        logp = np.loadtxt(path, usecols=(4, ))

        return cls(index=index, log_probabilities=logp)

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
        ind = self.index
        mol, sub, stage = ind['molecules'], ind['subensembles'], ind['stages']
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


class OrderParameterDistribution(Distribution):
    def __init__(self, index, log_probabilities):
        super(OrderParameterDistribution, self).__init__(index,
                                                         log_probabilities)

    @classmethod
    def from_file(cls, path):
        """Read the logarithm of the order parameter probability
        distribution from an lnpi_op.dat file.

        Args:
            path: The location of the file.

        Returns:
            An OrderParameterDistribution object.
        """
        index, log_probabilities = np.loadtxt(path, usecols=(0, 1),
                                              unpack=True)

        return cls(index=index.astype('int'),
                   log_probabilities=log_probabilities)

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
        p = np.poly1d(np.polyfit(x[:-1], y, order))
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


class FrequencyDistribution(object):
    sample_file = None

    def __init__(self, index, samples):
        self.index = index
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

    def __iter__(self):
        for s in self.samples:
            yield s

    @classmethod
    def from_combination(cls, distributions):
        index = distributions[0].index
        samples = sum([d.samples for d in distributions])

        return cls(index=index, samples=samples)

    @classmethod
    def from_combined_runs(cls, path, runs):
        return cls.from_combination(
            [cls.from_file(os.path.join(path, r, cls.sample_file))
             for r in runs])

    @classmethod
    def from_file(cls, path):
        raise NotImplementedError


class GrowthExpandedFrequencyDistribution(FrequencyDistribution):
    sample_file = 'hits_tr.dat'

    def __init__(self, index, samples):
        super(GrowthExpandedFrequencyDistribution, self).__init__(index,
                                                                  samples)

    @classmethod
    def from_file(cls, path):
        index = _read_growth_expanded_index(path)
        samples = np.loadtxt(path, usecols=(4, ), dtype='int', unpack=True)

        return cls(index=index, samples=samples)

    def write(self, path):
        _write_growth_expanded_distribution(path, self.index, self.samples)


class OrderParameterFrequencyDistribution(FrequencyDistribution):
    sample_file = 'hits_op.dat'

    def __init__(self, index, samples):
        super(OrderParameterFrequencyDistribution, self).__init__(index,
                                                                  samples)

    @classmethod
    def from_file(cls, path):
        index, samples = np.loadtxt(path, usecols=(0, 1), dtype='int',
                                    unpack=True)

        return cls(index=index, samples=samples)

    def write(self, path):
        _write_order_parameter_distribution(path, self.index, self.samples)


class PropertyList(object):
    freq_class = None
    prop_file = None

    def __init__(self, index, properties):
        self.index = index
        self.properties = properties

    @classmethod
    def from_combination(cls, lists, frequencies):
        weighted_sums = np.sum([lst * freq.samples
                                for lst, freq in zip(lists, frequencies)],
                               axis=0)
        freq_sum = sum([f.samples for f in frequencies])
        freq_sum[freq_sum < 1] = 1.0

        return cls(index=lists[0].index, properties=weighted_sums / freq_sum)

    @classmethod
    def from_combined_runs(cls, path, runs):
        fd = cls.freq_class
        lists = [cls.from_file(os.path.join(path, r, cls.prop_file))
                 for r in runs]
        frequencies = [fd.from_file(os.path.join(path, r, fd.sample_file))
                       for r in runs]

        return cls.from_combination(lists, frequencies)

    @classmethod
    def from_file(cls, path):
        raise NotImplementedError


class OrderParameterPropertyList(PropertyList):
    freq_class = OrderParameterFrequencyDistribution
    prop_file = 'prop_op.dat'

    def __init__(self, index, properties):
        super(OrderParameterPropertyList, self).__init__(index, properties)
        self.index = index
        self.samples = properties

    @classmethod
    def from_file(cls, path):
        raw = np.transpose(np.loadtxt(path))

        return cls(index=raw[0].astype('int'), properties=raw[1:])

    def write(self, path):
        fmt = ['%.8d'] + len(self.properties) * ['%10.5g']
        np.savetxt(path, np.transpose(np.append(self.index, self.properties)),
                   fmt=fmt, delimiter='  ')


class GrowthExpandedPropertyList(PropertyList):
    freq_class = GrowthExpandedFrequencyDistribution
    prop_file = 'prop_tr.dat'

    def __init__(self, index, properties):
        super(GrowthExpandedPropertyList, self).__init__(index, properties)
        self.index = index
        self.samples = properties

    @classmethod
    def from_file(cls, path):
        return cls(index=_read_growth_expanded_index(path),
                   properties=np.transpose(np.loadtxt(path))[4:])

    def write(self, path):
        ind = self.index
        fmt = 4 * ['%.8d'] + len(self.properties) * ['%10.5g']
        arr = np.append(ind['number'], ind['subensembles'], ind['molecules'],
                        ind['stages'], self.properties)
        np.savetxt(path, np.transpose(arr), fmt=fmt, delimiter='  ')
