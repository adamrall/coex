"""Objects and functions for working with probability distributions and
related properties.

Internally, we often deal with the logarithm of the probability
distribution along a path of interest instead of the free energy
differences, which differ only by a minus sign.

In gchybrid, we refer to the logarithm of the order parameter
distribution as lnpi_op.dat, the logarithm of the growth expanded
ensemble distribution as lnpi_tr.dat, the logarithm of the exchange path
distribution as lnpi_ex.dat, and the logarithm of the regrowth path
distribution as lnpi_rg.dat.  Similarly, the number of samples and other
properties along each path are contained in the files hits_*.dat and
prop_*.dat, respectively, where the * matches the appropriate two-letter
suffix.

"""

import copy
import os.path

import numpy as np


class TransitionMatrix(object):
    """A base class for acceptance probability matrices.

    Attributes:
        index: A numpy array or dict describing the states in the
            matrix.
        fw_atts: An array with the number of forward transition attempts
            for each state.
        rev_atts: An array with the number of reverse transition
            attempts for each state.
        fw_probs: An array with the acceptance probability for forward
            transitions from each state.
        rev_probs: An array with the acceptance probability for forward
            transitions from each state.
    """

    matrix_file = None

    def __init__(self, index, fw_atts, rev_atts, fw_probs, rev_probs):
        self.index = index
        self.fw_atts = fw_atts
        self.rev_atts = rev_atts
        self.fw_probs = fw_probs
        self.rev_probs = rev_probs

    def __len__(self):
        return len(self.fw_atts)

    @classmethod
    def from_combination(cls, matrices):
        """Combine a set of transition matrices.

        Args:
            matrices: A list of TransitionMatrix objects to combine.

        Returns:
            A TransitionMatrix object with the combined data.
        """
        index = matrices[0].index
        fw_atts = sum(m.fw_atts for m in matrices)
        rev_atts = sum(m.rev_atts for m in matrices)
        fw_probs = sum(m.fw_atts * m.fw_probs for m in matrices) / fw_atts
        rev_probs = sum(m.rev_atts * m.rev_probs for m in matrices) / rev_atts
        fw_probs, rev_probs = np.nan_to_num(fw_probs), np.nan_to_num(rev_probs)

        return cls(index=index, fw_atts=fw_atts, rev_atts=rev_atts,
                   fw_probs=fw_probs, rev_probs=rev_probs)

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
            cls.from_file(os.path.join(path, r, cls.matrix_file))
            for r in runs)

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
            A boolean numpy array, where True denotes states which
            don't meet the sampling quality threshold.
        """
        fw, rev = self.fw_atts, self.rev_atts
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
    """An acceptance probability matrix for a growth expanded
    ensemble path.

    Attributes:
        index: A dict with the keys 'numbers', 'subensembles',
            'molecules', and 'stages' describing the states in the
            matrix.
        fw_atts: An array with the number of forward transition attempts
            for each state.
        rev_atts: An array with the number of reverse transition
            attempts for each state.
        fw_probs: An array with the acceptance probability for forward
            transitions from each state.
        rev_probs: An array with the acceptance probability for forward
            transitions from each state.
    """

    matrix_file = 'pacc_tr_cr.dat'

    def __init__(self, index, fw_atts, rev_atts, fw_probs, rev_probs):
        super(GrowthExpandedTransitionMatrix, self).__init__(
            index, fw_atts, rev_atts, fw_probs, rev_probs)

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

        fw_att, rev_att = self.fw_atts, self.rev_atts
        fw_prob = self.fw_probs
        rev_prob = self.rev_probs
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

        return GrowthExpandedDistribution(index=ind, log_probs=dist)

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
        fw_atts, rev_atts, fw_probs, rev_probs = np.loadtxt(
            path, usecols=(4, 5, 6, 7), unpack=True)

        return cls(index, fw_atts=fw_atts.astype('int'),
                   rev_atts=rev_atts.astype('int'), fw_probs=fw_probs,
                   rev_probs=rev_probs)

    def write(self, path):
        """Write the transition matrix to a file.

        Args:
            path: The location of the file to write.
        """
        ind = self.index
        fmt = 6 * ['%8d'] + 2 * ['%10.5g']
        arr = np.column_stack((ind['number'], ind['subensembles'],
                               ind['molecules'], ind['stages'], self.fw_atts,
                               self.rev_atts, self.fw_probs, self.rev_probs))
        np.savetxt(path, arr, fmt=fmt, delimiter=' ')


class OrderParameterTransitionMatrix(TransitionMatrix):
    """An acceptance probability matrix for the order parameter path.

    Attributes:
        index: An array with the subensemble numbers.
        fw_atts: An array with the number of forward transition attempts
            for each state.
        rev_atts: An array with the number of reverse transition
            attempts for each state.
        fw_probs: An array with the acceptance probability for forward
            transitions from each state.
        rev_probs: An array with the acceptance probability for forward
            transitions from each state.
    """

    matrix_file = 'pacc_op_cr.dat'

    def __init__(self, index, fw_atts, rev_atts, fw_probs, rev_probs):
        super(OrderParameterTransitionMatrix, self).__init__(
            index, fw_atts, rev_atts, fw_probs, rev_probs)

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
            fw_prob = self.fw_probs[i]
            rev_prob = self.rev_probs[i + 1]
            if (self.fw_atts[i] > min_attempts and
                    self.rev_atts[i + 1] > min_attempts and
                    fw_prob > 0.0 and rev_prob > 0.0):
                dist[i + 1] += np.log(fw_prob / rev_prob)

        return OrderParameterDistribution(index=self.index,
                                          log_probs=dist)

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
        index, fw_atts, rev_atts = [c.astype('int') for c in raw[:3]]
        fw_probs, rev_probs = raw[3:]

        return cls(index, fw_atts=fw_atts, rev_atts=rev_atts,
                   fw_probs=fw_probs, rev_probs=rev_probs)

    def write(self, path):
        """Write the transition matrix to a file.

        Args:
            path: The location of the file to write.
        """
        fmt = 3 * ['%8d'] + 2 * ['%10.5g']
        arr = np.column_stack((self.index, self.fw_atts, self.rev_atts,
                               self.fw_probs, self.rev_probs))
        np.savetxt(path, arr, fmt=fmt)


def _write_growth_expanded_distribution(path, index, probs, fmt=None):
    if fmt is None:
        fmt = 4 * ['%8d'] + ['%10.5g']

    arr = np.column_stack((index['numbers'], index['subensembles'],
                           index['molecules'], index['stages'], probs))
    np.savetxt(path, arr, fmt=fmt)


def _write_order_parameter_distribution(path, index, probs, fmt=None):
    if fmt is None:
        fmt = ['%8d', '%10.5g']

    np.savetxt(path, np.column_stack((index, probs)), fmt=fmt)


class Distribution(object):
    """A base class for log probability distributions.

    Attributes:
        index: A numpy array or dict describing the states in the
            distribution's path.
        log_probs: An array with the logarithm of the
            probability distribution.
    """

    def __init__(self, index, log_probs):
        self.index = index
        self.log_probs = log_probs

    def __add__(self, other):
        if isinstance(other, Distribution):
            return self.log_probs + other.log_probs
        elif isinstance(other, (np.ndarray, int, float)):
            return self.log_probs + other
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Distribution):
            return self.log_probs - other.log_probs
        elif isinstance(other, (np.ndarray, int, float)):
            return self.log_probs - other
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Distribution):
            return self.log_probs * other.log_probs
        elif isinstance(other, (np.ndarray, int, float)):
            return self.log_probs * other
        else:
            return NotImplemented

    def __div__(self, other):
        if isinstance(other, Distribution):
            return self.log_probs / other.log_probs
        elif isinstance(other, (np.ndarray, int, float)):
            return self.log_probs / other
        else:
            return NotImplemented

    def __len__(self):
        return len(self.log_probs)

    def __getitem__(self, i):
        return self.log_probs[i]

    def __setitem__(self, i, value):
        self.log_probs[i] = value

    def __iter__(self):
        for p in self.log_probs:
            yield p


class GrowthExpandedDistribution(Distribution):
    """The logarithm of the probability distribution along a growth
    expanded ensemble path.

    Attributes:
        index: A dict with the keys 'numbers', 'subensembles',
            'molecules', and 'stages' describing the states in the path.
        log_probs: A numpy array with the logarithm of the probability
            distribution.
    """

    def __init__(self, index, log_probs):
        super(GrowthExpandedDistribution, self).__init__(index, log_probs)

    @classmethod
    def from_file(cls, path):
        """Read the logarithm of the growth expanded ensemble
        probability distribution from an lnpi_tr.dat file.

        Args:
            path: The location of the file.

        Returns:
            A GrowthExpandedDistribution object.
        """
        index = _read_growth_expanded_index(path)
        logp = np.loadtxt(path, usecols=(4, ))

        return cls(index=index, log_probs=logp)

    def smooth(self, order, drop=None):
        """Perform curve fitting on the growth expanded ensemble free
        energy differences to produce a new estimate of the free energy.

        Args:
            order: The order of the polynomial used to fit the free
                energy differences.
            drop: A boolean numpy array denoting whether to drop each
                entry prior to fitting.

        Returns:
            A dict containing the index, molecule number, stage number,
            and new estimate for the free energy of each entry in the
            expanded ensemble growth path.
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
        smoothed.log_probs = dist

        return smoothed

    def write(self, path):
        """Write the new free energy to a file.

        Args:
            path: The file to write.
        """
        _write_growth_expanded_distribution(path, self.index,
                                            self.log_probs)


class OrderParameterDistribution(Distribution):
    """The logarithm of the probability distribution along a growth
    expanded ensemble path.

    Attributes:
        index: A numpy array with the list of order parameter values.
        log_probs: A numpy array with the logarithm of the
            probability distribution.
    """

    def __init__(self, index, log_probs):
        super(OrderParameterDistribution, self).__init__(index, log_probs)

    @classmethod
    def from_file(cls, path):
        """Read the logarithm of the order parameter probability
        distribution from an lnpi_op.dat file.

        Args:
            path: The location of the file.

        Returns:
            An OrderParameterDistribution object.
        """
        index, log_probs = np.loadtxt(path, usecols=(0, 1), unpack=True)

        return cls(index=index.astype('int'), log_probs=log_probs)

    def smooth(self, order, drop=None):
        """Perform curve fitting on the free energy differences to
        produce a new estimate of the free energy.

        Args:
            order: The order of the polynomial used to fit the free
                energy differences.
            drop: A boolean numpy array denoting whether to drop each
                subensemble prior to fitting.

        Returns:
            A dict containing the subensemble index and the new estimate
            for the free energy of the order parameter path.
        """
        if drop is None:
            drop = np.tile(False, len(self) - 1)

        x = self.index[~drop]
        y = np.diff(self[~drop])
        p = np.poly1d(np.polyfit(x[:-1], y, order))
        smoothed = np.append(0.0, np.cumsum(p(self.index[1:])))

        return OrderParameterDistribution(index=self.index, log_probs=smoothed)

    def split(self, split=0.5):
        """Split a distribution into two parts.

        Args:
            split: The fraction of the length to use as the boundary for
                the two parts.

        Returns:
            A tuple of OrderParameterDistribution objects.
        """
        bound = int(split * len(self))
        ind, logp = self.index, self.log_probs
        fst = OrderParameterDistribution(ind[:bound], logp[:bound])
        snd = OrderParameterDistribution(ind[bound:], logp[bound:])

        return fst, snd

    def transform(self, amount):
        """Perform linear transformation on a probability distribution.

        Args:
            order_param: The order parameter values.
            lnpi: The logarithm of the probabilities.
            amount: The amount to shift the distribution.

        Returns:
            A new OrderParameterDistribution with transformed log
            probabilities.
        """
        transformed = self.log_probs + amount * self.index

        return OrderParameterDistribution(index=self.index,
                                          log_probs=transformed)

    def write(self, path):
        """Write the new free energy to a file.

        Args:
            path: The file to write.
        """
        _write_order_parameter_distribution(path, self.index, self.log_probs)


class FrequencyDistribution(object):
    """A base class for frequency distributions.

    Attributes:
        index: A numpy array or dict describing the states in the
            distribution.
        samples: An array with the number of times each state was
            visited in the simulation.
    """

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
        """Combine a list of frequency distributions.

        Args:
            distributions: The list of FrequencyDistribution objects to
                combine.

        Returns:
            A new FrequencyDistribution object with the combined data.
        """
        index = distributions[0].index
        samples = sum([d.samples for d in distributions])

        return cls(index=index, samples=samples)

    @classmethod
    def from_combined_runs(cls, path, runs):
        """Combine a frequency distribution across a series of
        production runs.

        Args:
            path: The directory containing the production runs.
            runs: The list of runs to combine.

        Returns:
            A FrequencyDistribution object with the combined data.
        """
        return cls.from_combination(
            [cls.from_file(os.path.join(path, r, cls.sample_file))
             for r in runs])

    @classmethod
    def from_file(cls, path):
        raise NotImplementedError


class GrowthExpandedFrequencyDistribution(FrequencyDistribution):
    """The frequency distribution along a growth expanded ensemble path.

    Attributes:
        index: A dict with the keys 'numbers', 'subensembles',
            'molecules', and 'stages' describing the states in the path.
        samples: An array with the number of times each state was
            visited in the simulation.
    """

    sample_file = 'hits_tr.dat'

    def __init__(self, index, samples):
        super(GrowthExpandedFrequencyDistribution, self).__init__(index,
                                                                  samples)

    @classmethod
    def from_file(cls, path):
        """Read a frequency distribution from a hits_tr.dat file.

        Args:
            path: The location of the file to read.

        Returns:
            A GrowthExpandedFrequencyDistribution object.
        """
        index = _read_growth_expanded_index(path)
        samples = np.loadtxt(path, usecols=(4, ), dtype='int', unpack=True)

        return cls(index=index, samples=samples)

    def write(self, path):
        """Write the frequency distribution to a file.

        Args:
            path: The location of the file to write.
        """
        _write_growth_expanded_distribution(path, self.index, self.samples,
                                            fmt=4 * ['%8d'] + ['%10d'])


class OrderParameterFrequencyDistribution(FrequencyDistribution):
    """The frequency distribution along an order parameter path.

    Attributes:
        index: A numpy array with the subensemble values.
        samples: An array with the number of times each state was
            visited in the simulation.
    """

    sample_file = 'hits_op.dat'

    def __init__(self, index, samples):
        super(OrderParameterFrequencyDistribution, self).__init__(index,
                                                                  samples)

    @classmethod
    def from_file(cls, path):
        """Read a frequency distribution from a hits_op.dat file.

        Args:
            path: The location of the file to read.

        Returns:
            An OrderParameterFrequencyDistribution object.
        """
        index, samples = np.loadtxt(path, usecols=(0, 1), dtype='int',
                                    unpack=True)

        return cls(index=index, samples=samples)

    def write(self, path):
        """Write the frequency distribution to a file.

        Args:
            path: The location of the file to write.
        """
        _write_order_parameter_distribution(path, self.index, self.samples,
                                            fmt=['%8d', '%10d'])


class PropertyList(object):
    """A base class for average property lists, i.e., the data stored
    in prop_*.dat files.

    Attributes:
        index: An array or dict describing the states in the path.
        properties: A 2D array with the average properties along the
            path.
    """

    freq_class = None
    prop_file = None

    def __init__(self, index, properties):
        self.index = index
        self.properties = properties

    @classmethod
    def from_combination(cls, lists, frequencies):
        """Combine a set of average property lists.

        Args:
            lists: A list of PropertyList objects to combine.
            frequencies: A list of frequency distributions for the
                relevant path.

        Returns:
            A PropertyList object with the combined data.
        """
        weighted_sums = np.sum([lst * freq.samples
                                for lst, freq in zip(lists, frequencies)],
                               axis=0)
        freq_sum = sum([f.samples for f in frequencies])
        freq_sum[freq_sum < 1] = 1.0

        return cls(index=lists[0].index, properties=weighted_sums / freq_sum)

    @classmethod
    def from_combined_runs(cls, path, runs):
        """Combine an average property list across a series of
        production runs.

        Args:
            path: The directory containing the production runs.
            runs: The list of runs to combine.

        Returns:
            A PropertyList object with the combined data.
        """
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
    """A list of the average energy, standard deviation of the energy,
    etc. along the order parameter path.

    Attributes:
        index: An array with the order parameter values.
        properties: A 2D array with the average properties along the
            path.
    """

    freq_class = OrderParameterFrequencyDistribution
    prop_file = 'prop_op.dat'

    def __init__(self, index, properties):
        super(OrderParameterPropertyList, self).__init__(index, properties)
        self.index = index
        self.samples = properties

    @classmethod
    def from_file(cls, path):
        """Read a prop_op.dat file.

        Args:
            path: The location of the file to read.

        Returns:
            An OrderParameterPropertyList object.
        """
        raw = np.transpose(np.loadtxt(path))

        return cls(index=raw[0].astype('int'), properties=raw[1:])

    def write(self, path):
        """Write the average properties to a file.

        Args:
            path: The location of the file to write
        """
        fmt = ['%8d'] + len(self.properties) * ['%10.5g']
        np.savetxt(path, np.column_stack((self.index,
                                          np.tranpose(self.properties))),
                   fmt=fmt)


class GrowthExpandedPropertyList(PropertyList):
    """A list of the average energy, standard deviation of the
    energy, etc. along a growth expanded ensemble path.

    Attributes:
        index: A dict with the keys 'numbers', 'subensembles',
            'molecules', and 'stages' describing the states in the path.
        properties: A 2D array with the average properties along the
            path.
    """

    freq_class = GrowthExpandedFrequencyDistribution
    prop_file = 'prop_tr.dat'

    def __init__(self, index, properties):
        super(GrowthExpandedPropertyList, self).__init__(index, properties)
        self.index = index
        self.samples = properties

    @classmethod
    def from_file(cls, path):
        """Read a prop_tr.dat file.

        Args:
            path: The location of the file to read.

        Returns:
            A GrowthExpandedPropertyList object.
        """
        return cls(index=_read_growth_expanded_index(path),
                   properties=np.transpose(np.loadtxt(path))[4:])

    def write(self, path):
        """Write the average properties to a file.

        Args:
            path: The location of the file to write
        """
        ind = self.index
        fmt = 4 * ['%8d'] + len(self.properties) * ['%10.5g']
        arr = np.column_stack((ind['number'], ind['subensembles'],
                               ind['molecules'], ind['stages'],
                               np.transpose(self.properties)))
        np.savetxt(path, arr, fmt=fmt)


def _generic_reader(op_cls, tr_cls, path):
    base = os.path.basename(path)
    if 'op' in base:
        return op_cls.from_file(path)
    elif 'tr' in base:
        return tr_cls.from_file(path)
    else:
        return NotImplementedError


def read_prop(path):
    """Read a prop_op.dat or prop_tr.dat file.

    Args:
        path: The location of the file to read.

    Returns:
        An OrderParameterPropertyList or GrowthExpandedPropertyList
        object, based on the name of the file.
    """
    return _generic_reader(op_cls=OrderParameterPropertyList,
                           tr_cls=GrowthExpandedPropertyList, path=path)


def read_lnpi(path):
    """Read an lnpi_op.dat file or an lnpi_tr.dat file.

    Args:
        path: The location of the file to read.

    Returns:
        An OrderParameterDistribution or GrowthExpandedDistribution
        object, based on the name of the file.
    """
    return _generic_reader(op_cls=OrderParameterDistribution,
                           tr_cls=GrowthExpandedDistribution, path=path)


def read_pacc(path):
    """Read a pacc_op_*.dat file or a pacc_tr_*.dat file.

    Args:
        path: The location of the file to read.

    Returns:
        An OrderParameterTransitionMatrix or
        GrowthExpandedTransitionMatrix object, based on the name of
        the file.
    """
    return _generic_reader(op_cls=OrderParameterTransitionMatrix,
                           tr_cls=GrowthExpandedTransitionMatrix, path=path)


def read_hits(path):
    """Read a hits_op.dat or hits_tr.dat file.

    Args:
        path: The location of the file to read.

    Returns:
        An OrderParameterFrequencyDistribution or
        GrowthExpandedFrequencyDistribution object, based on the
        name of the file.
    """
    return _generic_reader(op_cls=OrderParameterFrequencyDistribution,
                           tr_cls=GrowthExpandedFrequencyDistribution,
                           path=path)
