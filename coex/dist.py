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
    """A base class for an acceptance probability matrix along a
    specified path.

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


class OrderParamTransitionMatrix(TransitionMatrix):
    """An acceptance probability matrix along the order parameter path.

    Attributes:
        index: A numpy array with the order parameter values.
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
        super().__init__(index, fw_atts, rev_atts, fw_probs, rev_probs)

    def calculate_lnpi_op(self, guess, min_attempts=1):
        """Calculate the free energy of the order parameter path.

        Args:
            guess: A numpy array or OrderParamDistribution with an
                initial guess for the free energy.
            min_attempts: The minimum number of transitions in each
                direction required to consider the transition matrix
                when updating the free energy estimate.

        Returns:
            An OrderParamDistribution.
        """
        dist = np.zeros(len(self))
        for i, dc in enumerate(np.diff(guess)):
            dist[i + 1] = dist[i] + dc
            fw_prob = self.fw_probs[i]
            rev_prob = self.rev_probs[i + 1]
            if (self.fw_atts[i] > min_attempts and
                    self.rev_atts[i + 1] > min_attempts and fw_prob > 0.0 and
                    rev_prob > 0.0):
                dist[i + 1] += np.log(fw_prob / rev_prob)

        return OrderParamDistribution(index=self.index, log_probs=dist)

    def write(self, path):
        """Write the transition matrix to a file.

        Args:
            path: The location of the file to write.
        """
        fmt = 3 * ['%8d'] + 2 * ['%.11e']
        arr = np.column_stack((self.index, self.fw_atts, self.rev_atts,
                               self.fw_probs, self.rev_probs))

        np.savetxt(path, arr, fmt=fmt, delimiter=' ')


class TransferTransitionMatrix(TransitionMatrix):
    """An acceptance probability matrix along the molecule transfer
    path.

    Attributes:
        index: A dict with the overall number, molecule, subensemble,
            and growth stage of each state in the path.
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
        super().__init__(index, fw_atts, rev_atts, fw_probs, rev_probs)

    def calculate_lnpi_tr(self, guess, min_attempts=1):
        """Calculate the free energy of the order parameter path.

        Args:
            guess: A numpy array or TransferDistribution with an
                initial guess for the free energy along the molecule
                transfer path.
            min_attempts: The minimum number of transitions in each
                direction required to consider the transition matrix
                when updating the free energy estimate.

        Returns:
            A TransferDistribution.
        """
        dist = np.zeros(len(self))
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
                    if (self.fw_atts[cs] > min_attempts and
                            self.rev_atts[ns] > min_attempts and
                            self.fw_probs[cs] > 0.0 and
                            self.rev_probs[ns] > 0.0):
                        dist[cs] -= np.log(self.fw_probs[cs] /
                                           self.rev_probs[ns])

        return TransferDistribution(index=ind, log_probs=dist)

    def calculate_lnpi_op(self, tr_guess, op_guess, species=1, min_attempts=1):
        """Calculate the free energy of the order parameter path using
        the transfer path of the order parameter species.

        This method is only applicable for direct simulations.

        Args:
            tr_guess: A numpy array or TransferDistribution with an
                initial guess for the free energy along the molecule
                transfer path.
            op_guess: A numpy array or OrderParamDistribution with an
                initial guess for the free energy along the order
                parameter path.
            species: The order parameter species.
            min_attempts: The minimum number of transitions in each
                direction required to consider the transition matrix
                when updating the free energy estimate.

        Returns:
            An OrderParamDistribution.
        """
        ind = self.index
        mol, sub, stages = ind['molecules'], ind['subensembles'], ind['stages']
        uniq_sub = np.unique(sub)
        dist = np.zeros(len(uniq_sub))
        lnpi_tr = self.calculate_lnpi_tr(tr_guess)
        for i in uniq_sub[1:]:
            sampled = True
            fsub = (mol == species) & (sub == i - 1)
            rsub = (mol == species) & (sub == i)
            fs = fsub & (stages == np.amax(stages[fsub]))
            rs = rsub & (stages == np.amin(stages[rsub]))
            diff = tr_guess[rs] - tr_guess[fs]
            if (self.fw_atts[fs] > min_attempts and
                    self.rev_atts[rs] > min_attempts and
                    self.fw_probs[fs] > 0.0 and
                    self.rev_probs[rs] > 0.0):
                diff += np.log(self.fw_probs[fs] / self.rev_probs[rs])
            else:
                sampled = False

            for m in stages[rsub][1:]:
                lm = rsub & (stages == m - 1)
                cm = rsub & (stages == m)
                if (self.fw_atts[lm] > min_attempts and
                        self.rev_atts[cm] > min_attempts and sampled):
                    diff += lnpi_tr[cm] - lnpi_tr[lm]
                else:
                    sampled = False

            if sampled:
                dist[i] = dist[i - 1] + diff
            else:
                dist[i] = dist[i - 1] + op_guess[i] - op_guess[i - 1]

        return OrderParamDistribution(index=uniq_sub, log_probs=dist)

    def write(self, path):
        """Write the transition matrix to a file.

        Args:
            path: The location of the file to write.
        """
        ind = self.index
        fmt = 6 * ['%8d'] + 2 * ['%.11e']
        arr = np.column_stack((
            ind['number'], ind['subensembles'], ind['molecules'],
            ind['stages'], self.fw_atts, self.rev_atts, self.fw_probs,
            self.rev_probs))

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
        A TransitionMatrix object.
    """
    base = os.path.basename(path)
    if 'tr' in base:
        index = _read_tr_index(path)
        rev_atts, fw_atts, rev_probs, fw_probs = np.loadtxt(
            path, usecols=(4, 5, 6, 7), unpack=True)
        return TransferTransitionMatrix(
            index, fw_atts=fw_atts.astype('int'),
            rev_atts=rev_atts.astype('int'), fw_probs=fw_probs,
            rev_probs=rev_probs)
    elif 'op' in base:
        index, rev_atts, fw_atts, rev_probs, fw_probs = np.loadtxt(
            path, usecols=(0, 1, 2, 3, 4), unpack=True)
        index = index.astype('int')
        return OrderParamTransitionMatrix(
            index, fw_atts=fw_atts.astype('int'),
            rev_atts=rev_atts.astype('int'), fw_probs=fw_probs,
            rev_probs=rev_probs)
    else:
        raise NotImplementedError


def combine_matrices(matrices):
    """Combine a set of transition matrices.

    Args:
        matrices: A list of TransitionMatrix-like objects to combine.

    Returns:
        An instance of an appropriate subclass of TransitionMatrix with
        the combined data.
    """
    index = matrices[0].index
    fw_atts = sum(m.fw_atts for m in matrices)
    rev_atts = sum(m.rev_atts for m in matrices)
    fw_probs = sum(m.fw_atts * m.fw_probs for m in matrices) / fw_atts
    rev_probs = sum(m.rev_atts * m.rev_probs for m in matrices) / rev_atts
    fw_probs, rev_probs = np.nan_to_num(fw_probs), np.nan_to_num(rev_probs)
    if isinstance(matrices[0], TransferTransitionMatrix):
        return TransferTransitionMatrix(
            index=index, fw_atts=fw_atts, rev_atts=rev_atts,
            fw_probs=fw_probs, rev_probs=rev_probs)
    elif isinstance(matrices[0], OrderParamTransitionMatrix):
        return OrderParamTransitionMatrix(
            index=index, fw_atts=fw_atts, rev_atts=rev_atts,
            fw_probs=fw_probs, rev_probs=rev_probs)
    else:
        raise NotImplementedError


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
    """A base class for probability distributions.

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


class OrderParamDistribution(Distribution):
    """The logarithm of the probability distribution along the order
    parameter path.

    Attributes:
        index: A numpy array with the order parameter values.
        log_probs: An array with the logarithm of the
            probability distribution.
    """

    def __init__(self, index, log_probs):
        super().__init__(index, log_probs)

    def smooth(self, order, denominator=None, drop=None):
        """Perform curve fitting on the free energy differences to
        produce a new estimate of the free energy.

        Args:
            order: The order of the polynomial used to fit the free
                energy differences.
            denominator: If present, smooth the differences in free
                energy relative to this array. Useful for, e.g.,
                smoothing relative to beta in TEE simulations.
            drop: A boolean numpy array denoting whether to drop each
                subensemble prior to fitting.

        Returns:
            An OrderParamDistribution with the new estimate for the
            free energy.
        """
        if drop is None:
            drop = np.tile(False, len(self) - 1)
        else:
            drop = drop[1:]

        if denominator is None:
            denominator = np.array(range(len(self))) + 1.0

        x = self.index[1:][~drop]
        y = (np.diff(self) / np.diff(denominator))[~drop]
        p = np.poly1d(np.polyfit(x, y, order))
        smoothed = p(self.index) * denominator
        smoothed -= smoothed[0]

        return OrderParamDistribution(index=self.index, log_probs=smoothed)

    def split(self, split=0.5):
        """Split the distribution into two parts.

        Args:
            split: The fraction of the length to use as the boundary for
                the two parts.

        Returns:
            A tuple of OrderParamDistribution objects.
        """
        bound = int(split * len(self))
        ind, logp = self.index, self.log_probs
        fst = OrderParamDistribution(index=ind[:bound],
                                     log_probs=logp[:bound])
        snd = OrderParamDistribution(index=ind[bound:],
                                     log_probs=logp[bound:])

        return fst, snd

    def write(self, path):
        """Write the distribution to a file.

        Args:
            path: The name of the file to write.
        """
        np.savetxt(path, np.column_stack((self.index, self.log_probs)),
                   fmt=['%8d', '%.11e'])


class TransferDistribution(Distribution):
    """The logarithm of the probability distribution along the
    molecule transfer path.

    Attributes:
        index: A dict with the overall number, molecule, subensemble,
            and growth stage of each state in the path.
        log_probs: An array with the logarithm of the
            probability distribution.
    """

    def __init__(self, index, log_probs):
        super().__init__(index, log_probs)

    def shift_by_order_parameter(self, op_dist):
        """Add the order parameter free energies to transfer path
        free energies.

        This is the form that gchybrid normally outputs: each
        subensemble's transfer path free energies are relative to that
        subensemble's order parameter free energy.

        Args:
            op_dist: An OrderParamDistribution object with the free
                energies to shift by.

        Returns:
            A new TransferDistribution with shifted free energies.
        """
        shifted = copy.deepcopy(self)
        for i, p in enumerate(op_dist):
            shifted[shifted.index['subensembles'] == i] += p

        return shifted

    def smooth(self, order, drop=None):
        """Perform curve fitting on the free energy differences to
        produce a new estimate of the free energy.

        Args:
            order: The order of the polynomial used to fit the free
                energy differences.
            drop: A boolean numpy array denoting whether to drop each
                subensemble prior to fitting.

        Returns:
            A TransferDistribution with the new estimate for the free
            energy.
        """
        size = len(self)
        ind = self.index
        mol, sub, stage = ind['molecules'], ind['subensembles'], ind['stages']
        diff, fit = np.zeros(size), np.zeros(size)
        dist = np.zeros(size)
        x = np.array(range(size))
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
                p = np.poly1d(np.polyfit(x[curr_stage & ~drop], y, order))
                fit[curr_stage] = p(x[curr_stage])

            for s in mol_subs:
                curr_sub = (sub == s)
                for i in reversed(mol_stages):
                    curr_stage = curr_mol & curr_sub & (stage == i)
                    next_stage = curr_mol & curr_sub & (stage == i + 1)
                    dist[curr_stage] = dist[next_stage] - fit[curr_stage]

        smoothed = copy.deepcopy(self)
        smoothed.log_probs = dist

        return smoothed

    def split(self, split=0.5):
        """Split the distribution into two parts.

        Args:
            split: The fraction of the length to use as the boundary for
                the two parts.

        Returns:
            A tuple of TransferDistribution objects.
        """
        bound = int(split * len(self))
        ind, logp = self.index, self.log_probs
        fst = TransferDistribution(
            index={k: v[:bound] for k, v in ind.items()},
            log_probs=logp[:bound])
        snd = TransferDistribution(
            index={k: v[bound:] for k, v in ind.items()},
            log_probs=logp[bound:])

        return fst, snd

    def write(self, path):
        """Write the distribution to a file.

        Args:
            path: The name of the file to write.
        """
        ind = self.index
        np.savetxt(path, np.column_stack((ind['numbers'], ind['subensembles'],
                                          ind['molecules'], ind['stages'],
                                          self.log_probs)),
                   fmt=4 * ['%8d'] + ['%.11e'])


def read_lnpi(path):
    """Read an lnpi_op.dat file or an lnpi_tr.dat file.

    Args:
        path: The location of the file to read.

    Returns:
        A Distribution object.
    """
    base = os.path.basename(path)
    if 'tr' in base:
        return TransferDistribution(index=_read_tr_index(path),
                                    log_probs=np.loadtxt(path, usecols=(4, )))
    elif 'op' in base:
        index, logp = np.loadtxt(path, usecols=(0, 1), unpack=True)

        return OrderParamDistribution(index=index.astype('int'),
                                      log_probs=logp)
    else:
        raise NotImplementedError


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
                fmt=4 * ['%8d'] + ['%.11e'])
        else:
                np.savetxt(path, np.column_stack((self.index, self.freqs)),
                           fmt=['%8d', '%.11e'])


def read_hits(path):
    """Read a hits_op.dat or hits_tr.dat file.

    Args:
        path: The location of the file to read.

    Returns:
        An FrequencyDistribution object.
    """
    base = os.path.basename(path)
    if 'tr' in base:
        return FrequencyDistribution(
            index=_read_tr_index(path),
            freqs=np.loadtxt(path, usecols=(4, ), dtype='int', unpack=True))
    elif 'op' in base:
        index, freqs = np.loadtxt(path, usecols=(0, 1), dtype='int',
                                  unpack=True)

        return FrequencyDistribution(index=index, freqs=freqs)
    else:
        raise NotImplementedError


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
                fmt=4 * ['%8d'] + len(self.props) * ['%.11e'])
        else:
            np.savetxt(path, np.column_stack((self.index,
                                              np.tranpose(*self.props))),
                       fmt=['%8d'] + len(self.props) * ['%.11e'])


def read_prop(path):
    """Read a prop_op.dat or prop_tr.dat file.

    Args:
        path: The location of the file to read.

    Returns:
        A PropertyList object.
    """
    base = os.path.basename(path)
    if 'tr' in base:
        return PropertyList(index=_read_tr_index(path),
                            props=np.transpose(np.loadtxt(path))[4:])
    elif 'op' in base:
        raw = np.transpose(np.loadtxt(path))

        return PropertyList(index=raw[0].astype('int'), props=raw[1:])
    else:
        raise NotImplementedError


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
