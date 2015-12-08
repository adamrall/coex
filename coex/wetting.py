# wetting.py
# Copyright (C) 2015 Adam R. Rall <arall@buffalo.edu>
#
# This file is part of coex.
#
# coex is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# coex is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with coex.  If not, see <http://www.gnu.org/licenses/>.

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
    """Calculate the change in spreading/drying coefficient for a pair of
    simulations.

    Args:
        valley: The interface potential of the valley region.
        plateau: The interface potential of the plateau region.
        index: The reference subensemble number.
        reference: The reference spreading/drying coefficient.

    Returns:
        A numpy array with the spreading/drying coefficient of each
        subensemble.

    See Also:
        interface_potential() for a description of the interface
        potential.
    """
    return reference + (valley - valley[index]) - (plateau - plateau[index])


def interface_potential(dist, area, beta):
    """Convert a logarithmic probability distribution to an interface
    potential.

    The interface potential is the free energy required to form a
    fluid film of a specified thickness.  In a partially wetting
    system (i.e., one that would form a bubble/droplet of one phase on
    a surface surrounded by another phase), the interface potential in
    a direct simulation has a global minimum at near-zero fluid
    thickness and a plateau for the formation of a thick film.  We use
    the difference between these two regions to find a measure of the
    spreading or drying coefficient of the system.  In an expanded
    ensemble simulation, we track the two regions separately;
    differences in the interface potential of each region can be
    combined to find differences in the spreading or drying
    coefficient with respect to some expanded ensemble path.

    Args:

        dist: A distribution, i.e., a dict with keys 'param' and
            'logp'.
        area: The x*y area of the simulation in m^2.
        beta: The thermodynamic beta (1/kT) of the simulation.

    Returns:
        A numpy array with the interface potential.

    See Also:
        read.read_lnpi() for the structure of the distribution.

    """
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
