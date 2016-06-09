"""Find the wetting properties of a direct or expanded ensemble
grand canonical simulation.
"""

import numpy as np


def get_cos_theta(s, d):
    """Calculate the cosine of the contact angle.

    Args:
        s: A float (or numpy array): the spreading coefficient.
        d: A float (or numpy array): the drying coefficient.

    Returns:
        The cosine of the contact angle as a float or numpy array.
    """
    return -(s - d) / (s + d)


def get_drying_coefficient(dist):
    """Calculate the drying coefficient.

    Args:
        dist: A Distribution object from a direct (GC) drying
            simulation.

    Returns:
        The dimensionless drying coefficient (beta*d*A).
    """
    potential = -dist.log_probs
    valley = np.amin(potential)
    split = int(0.5 * len(potential))
    plateau = np.mean(potential[:split])

    return valley - plateau


def get_expanded_ensemble_coefficients(valley, plateau, index, reference):
    """Calculate the change in spreading/drying coefficient for a pair of
    simulations.

    Args:
        valley: A Distribution from the valley region.
        plateau: A Distribution from the plateau region.
        index: The reference subensemble number.
        reference: The reference spreading/drying coefficient.

    Returns:
        A numpy array with the spreading/drying coefficient of each
        subensemble.
    """
    return reference - (valley - valley[index]) + (plateau - plateau[index])


def get_spreading_coefficient(dist):
    """Calculate the spreading coefficient.

    Args:
        dist: A Distribution from a direct (GC) spreading simulation.

    Returns:
        The dimensionless spreading coefficient (beta*s*A).
    """
    potential = -dist.log_probs
    valley = np.amin(potential)
    split = int(0.5 * len(potential))
    plateau = np.mean(potential[split:])

    return valley - plateau


def get_tension(s, d):
    """Calculate the interfacial tension.

    Args:
        s: A float (or numpy array): the spreading coefficient.
        d: A float (or numpy array): the drying coefficient.

    Returns:
        The interfacial tension in the appropriate units.
    """
    return -0.5 * (s + d)
