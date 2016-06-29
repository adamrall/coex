"""Functions dealing with activities and activity fractions."""

import numpy as np


def activities_to_fractions(activities, one_subensemble=False):
    """Convert a list of activities to activity fractions.

    Args:
        activities: A numpy array with the activities of the system.
        one_subensemble: A bool that describes the shape of the
            input/output.

    Returns:
        A numpy array with the logarithm of the sum of the activities
        and the activity fractions of each species after the first.
        If the array is multidimensional, each column corresponds to
        a subensemble from an expanded ensemble simulation.
    """
    if isinstance(activities, list):
        activities = np.array(activities)

    if ((not one_subensemble and len(activities.shape) == 1) or
            (one_subensemble and len(activities) == 1)):
        return np.log(activities)

    fractions = np.copy(activities)
    fractions[0] = np.log(sum(activities))
    fractions[1:] /= np.exp(fractions[0])

    return fractions


def fractions_to_activities(fractions, one_subensemble=False):
    """Convert a list of activity fractions to activities.

    Args:
        fractions: A numpy array with the activity fractions.
        one_subensemble: A bool that describes the shape of the
            input/output.

    Returns:
        A numpy array with the activities. If the array is
        multidimensional, each column corresponds to a subensemble
        from an expanded ensemble simulation.
    """
    if isinstance(fractions, list):
        fractions = np.array(fractions)

    if ((not one_subensemble and len(fractions.shape) == 1) or
            (one_subensemble and len(fractions) == 1)):
        return np.exp(fractions)

    activities = np.copy(fractions)
    activity_sum = np.exp(fractions[0])
    activities[1:] *= activity_sum
    activities[0] = activity_sum - sum(activities[1:])

    return activities


def read_zz(path):
    """Read the activity fractions of an AFEE simulation.

    Args:
        path: The location of the 'zz.dat' file to read.

    Returns:
        A numpy array: the first row contains the logarithm of the sum
        of the activities for each subensemble, and each subsequent
        row contains the activity fractions of each species after the
        first.
    """
    # Truncate the first column, which just contains an index, and
    # transpose the rest.
    return np.transpose(np.loadtxt(path))[1:]


def write_zz(path, fractions):
    """Write the activity fractions to file.

    Args:
        path: The 'zz.dat' file to write.
        fractions: A numpy array with activity fractions.
    """
    cols = fractions.shape[1]
    arr = np.column_stack((range(cols), *fractions))
    np.savetxt(path, arr, fmt=['%d'] + cols * ['%.15g'])


def read_bz(path):
    """Read the activity fractions and beta values of a TEE simulation.

    Args:
        path: The location of the 'bz.dat' file to read.

    Returns:
        A dict containing numpy arrays of beta and the activity
        fractions.
    """
    # Truncate the first column, which just contains an index, read
    # beta separately, and transpose the rest.
    beta = np.loadtxt(path, usecols=(1, ))
    zz = np.transpose(np.loadtxt(path))[2:]

    return {'beta': beta, 'fractions': zz}


def write_bz(path, beta, fractions):
    """Write the activity fractions and beta values of a TEE simulation.

    Args:
        path: The 'bz.dat' file to write.
        beta: The list of inverse temperature (1/kT) values.
        fractions: A numpy array with the activity fractions.
    """
    arr = np.column_stack((range(len(beta)), beta, *fractions))
    np.savetxt(path, arr, fmt=['%d'] + (arr.shape[1] - 1) * ['%.15g'])
