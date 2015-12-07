# TODO: Finish writing documentation.
"""Find the wetting properties of a direct or expanded ensemble
grand canonical simulation.
"""

from __future__ import division

import numpy as np


def cos_theta(s, d):
    """Calculate the cosine of the contact angle.

    Args:
        s: A float (or numpy array): the spreading coefficient.
        d: A float (or numpy array): the drying coefficient.

    Returns:
        The cosine of the contact angle as a float or numpy array.
    """
    return -(s - d) / (s + d)


def drying_coefficient(potential):
    """Calculate the drying coefficient.

    Args:
        potential: The interface potential of the simulation.

    Returns:
        The drying coefficient in J/m^2.

    See also:
        interface_potential(), get_spreading_coefficient()
    """
    valley = np.amin(potential)
    split = int(0.5 * len(potential))
    plateau = np.mean(potential[:split])

    return valley - plateau


def expanded_ensemble_coefficients(valley, plateau, index, reference):
    return reference + (valley - valley[index]) - (plateau - plateau[index])


def interface_potential(dist, area, beta):
    return -dist['logp'] / area / beta


def spreading_coefficient(potential):
    """Calculate the spreading coefficient.

    Args:
        potential: The interface potential of the simulation.

    Returns:
        The spreading coefficient in J/m^2.

    See Also:
        interface_potential(), get_drying_coefficient()
    """
    valley = np.amin(potential)
    split = int(0.5 * len(potential))
    plateau = np.mean(potential[split:])

    return valley - plateau


def tension(s, d):
    """Calculate the interfacial tension.

    Args:
        s: A float (or numpy array): the spreading coefficient.
        d: A float (or numpy array): the drying coefficient.

    Returns:
        The interfacial tension in SI units.
    """
    return -0.5 * (s + d)
