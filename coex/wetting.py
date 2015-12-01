"""Find the wetting properties of a grand canonical simulation."""

from __future__ import division
import os.path

import numpy as np

from coex.read import read_lnpi


def read_potential(directory, area, beta):
    path = os.path.join(directory, 'lnpi_op.dat')

    return -read_lnpi(path)['logp'] / area / beta


# TODO: Add shifting to reference beta/activities.
def get_expanded_ensemble_coefficients(valley, plateau, reference):
    valley_potential = read_potential(valley, reference['area'],
                                      reference['beta'])
    plateau_potential = read_potential(plateau, reference['area'],
                                       reference['beta'])

    return reference['coefficient'] + valley_potential - plateau_potential


def get_spreading_coefficient(directory, area, beta):
    """Calculate the spreading coefficient.

    Args:
        directory: The location of the simulation.
        area: The area of the surface in the simulation in m^2.
        beta: The thermodynamic beta (1/kT) of the simulation.

    Returns:
        The spreading coefficient in J/m^2.
    """
    potential = read_potential(os.path.join(directory, 'lnpi_op.dat'), area,
                               beta)
    valley = np.amin(potential)
    split = int(0.5 * len(potential))
    plateau = np.mean(potential[split:])

    return valley - plateau


def get_drying_coefficient(directory, area, beta):
    """Calculate the drying coefficient.

    Args:
        directory: The location of the simulation.
        area: The area of the surface in the simulation in m^2.
        beta: The thermodynamic beta (1/kT) of the simulation.

    Returns:
        The drying coefficient in J/m^2.
    """
    potential = read_potential(os.path.join(directory, 'lnpi_op.dat'), area,
                               beta)
    valley = np.amin(potential)
    split = int(0.5 * len(potential))
    plateau = np.mean(potential[:split])

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


def cos_theta(s, d):
    """Calculate the cosine of the contact angle.

    Args:
        s: A float (or numpy array): the spreading coefficient.
        d: A float (or numpy array): the drying coefficient.

    Returns:
        The cosine of the contact angle as a float or numpy array.
    """
    return -(s - d) / (s + d)
