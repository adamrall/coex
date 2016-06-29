"""Objects and functions for working with probability distributions and
related properties.

Internally, we often deal with the logarithm of the probability
distribution along a path of interest instead of the free energy
differences, which differ only by a minus sign.

In gchybrid, we refer to the logarithm of the order parameter
distribution as lnpi_op.dat, the logarithm of the growth expanded
ensemble distribution as lnpi_tr.dat, the logarithm of the exchange path
distribution as lnpi_ex.dat, and the logarithm of the regrowth path
distribution as lnpi_rg.dat.  Similarly, the frequency distribution
along each path is contained in the file hits_*.dat, where
the * matches the appropriate two-letter suffix.
"""

import copy
import os.path

import numpy as np


class TransitionMatrix(object):
    """An acceptance probability matrix along a specified path.

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

    def __init__(self, index, fw_atts, rev_atts, fw_probs, rev_probs):
        self.index = index
        self.fw_atts = fw_atts
        self.rev_atts = rev_atts
        self.fw_probs = fw_probs
        self.rev_probs = rev_probs

    def __len__(self):
        return len(self.fw_atts)

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

    def _calculate_lnpi_tr(self, guess, min_attempts):
        dist = np.zeros(len(self))
        if guess is None:
            guess = np.copy(dist)

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
                    if (self.fw_att[cs] > min_attempts and
                            self.rev_att[ns] > min_attempts and
                            self.fw_prob[cs] > 0.0 and
                            self.rev_prob[ns] > 0.0):
                        dist[cs] -= np.log(self.fw_prob[cs] /
                                           self.rev_prob[ns])

        return Distribution(index=ind, log_probs=dist)

    def _calculate_lnpi_op(self, guess, min_attempts):
        dist = np.zeros(len(self))
        if guess is None:
            guess = np.copy(dist)

        for i, dc in enumerate(np.diff(guess)):
            dist[i + 1] = dist[i] + dc
            fw_prob = self.fw_probs[i]
            rev_prob = self.rev_probs[i + 1]
            if (self.fw_atts[i] > min_attempts and
                    self.rev_atts[i + 1] > min_attempts and fw_prob > 0.0 and
                    rev_prob > 0.0):
                dist[i + 1] += np.log(fw_prob / rev_prob)

        return Distribution(index=self.index, log_probs=dist)

    def calculate_lnpi(self, guess=None, min_attempts=1):
        """Compute the logarithm of the probability distribution using
        the transition matrix.

        Args:
            guess: An initial guess for the logarithm of the
                probability distribution.
            min_attempts: The threshold for considering a transition
                adequately sampled.

        Returns:
            A Distribution object with the computed logarithm of the
            probability distribution.
        """
        if isinstance(self.index, dict):
            return self._calculate_lnpi_tr(guess, min_attempts)

        return self._calculate_lnpi_op(guess, min_attempts)

    def write(self, path):
        """Write a transition matrix to a file.

        Args:
            path: The location of the file to write.
        """
        if isinstance(self.index, dict):
            ind = self.index
            fmt = 6 * ['%8d'] + 2 * ['%10.5g']
            arr = np.column_stack((
                ind['number'], ind['subensembles'], ind['molecules'],
                ind['stages'], self.fw_atts, self.rev_atts, self.fw_probs,
                self.rev_probs))
        else:
            fmt = 3 * ['%8d'] + 2 * ['%10.5g']
            arr = np.column_stack((self.index, self.fw_atts, self.rev_atts,
                                   self.fw_probs, self.rev_probs))

        np.savetxt(path, arr, fmt=fmt, delimiter=' ')


def _read_tr_index(path):
    num, sub, mol, stg = np.loadtxt(path, usecols=(0, 1, 2, 3),
                                    dtype='int', unpack=True)

    return {'numbers': num, 'subensembles': sub, 'molecules': mol,
            'stages': stg}


def read_pacc(path):
    """Read a pacc_op_*.dat file or a pacc_tr_*.dat file.

    Args:
        path: The location of the file to read.

    Returns:
        An TransitionMatrix object.
    """
    base = os.path.basename(path)
    if 'tr' in base:
        index = _read_tr_index(path)
        fw_atts, rev_atts, fw_probs, rev_probs = np.loadtxt(
            path, usecols=(4, 5, 6, 7), unpack=True)
    elif 'op' in base:
        index, fw_atts, rev_atts, fw_probs, rev_probs = np.loadtxt(
            path, usecols=(0, 1, 2, 3, 4), unpack=True)
        index = index.astype('int')
    else:
        raise NotImplementedError

    return TransitionMatrix(index, fw_atts=fw_atts.astype('int'),
                            rev_atts=rev_atts.astype('int'), fw_probs=fw_probs,
                            rev_probs=rev_probs)


def combine_matrices(matrices):
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

    return TransitionMatrix(index=index, fw_atts=fw_atts, rev_atts=rev_atts,
                            fw_probs=fw_probs, rev_probs=rev_probs)


def combine_pacc_runs(path, runs, pacc_file):
    """Combine a set of transition matrix files.

    Args:
        path: The base path containing the data to combine.
        runs: The list of runs to combine.
        pacc_file: The name of the file to combine.

    Returns:
        A TransitionMatrix object with the combined data.
    """
    return combine_matrices([read_pacc(os.path.join(path, r, pacc_file))
                             for r in runs])


class Distribution(object):
    """The logarithm of the probability distribution along a specified
    path.

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

    def _smooth_op(self, order, drop=None):
        if drop is None:
            drop = np.tile(False, len(self) - 1)

        x = self.index[~drop]
        y = np.diff(self[~drop])
        p = np.poly1d(np.polyfit(x[:-1], y, order))
        smoothed = np.append(0.0, np.cumsum(p(self.index[1:])))

        return Distribution(index=self.index, log_probs=smoothed)

    def _smooth_tr(self, order, drop=None):
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

        smoothed = copy.deepcopy(self)
        smoothed.log_probs = dist

        return smoothed

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
        if isinstance(self.index, dict):
            return self._smooth_tr(order, drop)

        return self._smooth_op(order, drop)

    def split(self, split=0.5):
        """Split the distribution into two parts.

        Args:
            split: The fraction of the length to use as the boundary for
                the two parts.

        Returns:
            A tuple of Distribution objects.
        """
        bound = int(split * len(self))
        ind, logp = self.index, self.log_probs
        if isinstance(ind, dict):
            fst = Distribution(index={k: v[:bound] for k, v in ind.items()},
                               log_probs=logp[:bound])
            snd = Distribution(index={k: v[bound:] for k, v in ind.items()},
                               log_probs=logp[bound:])
        else:
            fst = Distribution(index=ind[:bound], log_probs=logp[:bound])
            snd = Distribution(index=ind[bound:], log_probs=logp[bound:])

        return fst, snd

    def write(self, path):
        """Write the distribution to a file.

        Args:
            path: The name of the file to write.
        """
        if isinstance(self.index, dict):
            ind = self.index
            np.savetxt(
                path,
                np.column_stack((ind['numbers'], ind['subensembles'],
                                 ind['molecules'], ind['stages'],
                                 self.log_probs)),
                fmt=4 * ['%8d'] + ['%10.5g'])
        else:
                np.savetxt(path, np.column_stack((self.index, self.log_probs)),
                           fmt=['%8d', '%10.5g'])


def read_lnpi(path):
    """Read an lnpi_op.dat file or an lnpi_tr.dat file.

    Args:
        path: The location of the file to read.

    Returns:
        A Distribution object.
    """
    base = os.path.basename(path)
    if 'tr' in base:
        index = _read_tr_index(path)
        logp = np.loadtxt(path, usecols=(4, ))
    elif 'op' in base:
        index, logp = np.loadtxt(path, usecols=(0, 1), unpack=True)
        index = index.astype('int')
    else:
        raise NotImplementedError

    return Distribution(index=index, log_probs=logp)


class FrequencyDistribution(object):
    """The frequency distribution along a specified path.

    Attributes:
        index: A numpy array or dict describing the states in the
            distribution.
        freqs: An array with the number of times each state was
            visited in the simulation.
    """
    def __init__(self, index, freqs):
        self.index = index
        self.freqs = freqs

    def __len__(self):
        return len(self.freqs)

    def __getitem__(self, i):
        return self.freqs[i]

    def __iter__(self):
        for s in self.freqs:
            yield s

    def write(self, path):
        """Write the frequency distribution to a file.

        Args:
            path: The location of the file to write.
        """
        if isinstance(self.index, dict):
            ind = self.index
            np.savetxt(
                path,
                np.column_stack((ind['numbers'], ind['subensembles'],
                                 ind['molecules'], ind['stages'],
                                 self.freqs)),
                fmt=4 * ['%8d'] + ['%10.5g'])
        else:
                np.savetxt(path, np.column_stack((self.index, self.freqs)),
                           fmt=['%8d', '%10.5g'])


def read_hits(path):
    """Read a hits_op.dat or hits_tr.dat file.

    Args:
        path: The location of the file to read.

    Returns:
        An FrequencyDistribution object.
    """
    base = os.path.basename(path)
    if 'tr' in base:
        index = _read_tr_index(path)
        freqs = np.loadtxt(path, usecols=(4, ), dtype='int', unpack=True)
    elif 'op' in base:
        index, freqs = np.loadtxt(path, usecols=(0, 1), dtype='int',
                                  unpack=True)
    else:
        raise NotImplementedError

    return FrequencyDistribution(index=index, freqs=freqs)


def combine_frequencies(cls, dists):
    """Combine a list of frequency distributions.

    Args:
        distributions: The list of FrequencyDistribution objects to
            combine.

    Returns:
        A new FrequencyDistribution object with the combined data.
    """
    index = dists[0].index
    freqs = sum([d.freqs for d in dists])

    return FrequencyDistribution(index=index, freqs=freqs)


def combine_hits_runs(path, runs, hits_file):
    """Combine a frequency distribution across a series of
    production runs.

    Args:
        path: The directory containing the production runs.
        runs: The list of runs to combine.
        hits_file: The name of the file to combine.

    Returns:
        A FrequencyDistribution object with the combined data.
    """
    return combine_frequencies([read_hits(os.path.join(path, r, hits_file))
                                for r in runs])


class PropertyList(object):
    """An average property list, i.e., the data stored in prop_*.dat
    files.

    Attributes:
        index: An array or dict describing the states in the path.
        props: A 2D array with the average properties along the
            path.
    """

    def __init__(self, index, props):
        self.index = index
        self.props = props

    def write(self, path):
        """Write the average properties to a file.

        Args:
            path: The location of the file to write
        """
        if isinstance(self.index, dict):
            ind = self.index
            np.savetxt(
                path,
                np.column_stack((ind['number'], ind['subensembles'],
                                 ind['molecules'], ind['stages'],
                                 *np.transpose(self.props))),
                fmt=4 * ['%8d'] + len(self.props) * ['%10.5g'])
        else:
            np.savetxt(path, np.column_stack((self.index,
                                              np.tranpose(*self.props))),
                       fmt=['%8d'] + len(self.props) * ['%10.5g'])


def read_prop(path):
    """Read a prop_op.dat or prop_tr.dat file.

    Args:
        path: The location of the file to read.

    Returns:
        An OrderParameterPropertyList or GrowthExpandedPropertyList
        object, based on the name of the file.
    """
    base = os.path.basename(path)
    if 'tr' in base:
        index = _read_tr_index(path)
        props = np.transpose(np.loadtxt(path))[4:]
    elif 'op' in base:
        raw = np.transpose(np.loadtxt(path))
        index = raw[0].astype('int')
        props = raw[1:]
    else:
        raise NotImplementedError

    return PropertyList(index=index, props=props)


def combine_property_lists(cls, lists, freq_dists):
    """Combine a set of average property lists.

    Args:
        lists: A list of PropertyList objects to combine.
        freq_dists: A list of FrequencyDistribution objects for the
            relevant path.

    Returns:
        A PropertyList object with the combined data.
    """
    weighted_sums = np.sum([lst * fd.freqs
                            for lst, fd in zip(lists, freq_dists)], axis=0)
    freq_sum = sum([f.freqs for f in freq_dists])
    freq_sum[freq_sum < 1] = 1.0

    return PropertyList(index=lists[0].index, props=weighted_sums / freq_sum)


def combine_prop_runs(path, runs, prop_file):
    """Combine an average property list across a series of
    production runs.

    Args:
        path: The directory containing the production runs.
        runs: The list of runs to combine.
        prop_file: The name of the file to read.

    Returns:
        A PropertyList object with the combined data.
    """
    hits_file = prop_file.replace('prop', 'hits')

    lists = [read_prop(os.path.join(path, r, prop_file)) for r in runs]
    freq_dists = [read_hits(os.path.join(path, r, hits_file)) for r in runs]

    return combine_property_lists(lists, freq_dists)
