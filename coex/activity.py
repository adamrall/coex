# activity.py
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

"""Convert activity fractions to activities and vice versa."""

from __future__ import division

import numpy as np


def activities_to_fractions(activities, one_dimensional=False):
    """Convert a list of activities to activity fractions.

    Args:
        activities: A numpy array with the activities of the system.
        one_dimensional: A bool that describes the shape of the
            input/output.

    Returns:
        A numpy array with the logarithm of the sum of the activities
        and the activity fractions of each species after the first.
        If the array is multidimensional, each column corresponds to
        a subensemble from an expanded ensemble simulation.

    See Also:
        fractions_to_activities() for the opposite conversion.
    """
    if len(activities.shape) == 1 or (one_dimensional and len(activities) == 1):
        return np.log(activities)

    fractions = np.copy(activities)
    fractions[0] = np.log(sum(activities))
    fractions[1:] /= np.exp(fractions[0])

    return fractions


def fractions_to_activities(fractions, one_dimensional=False):
    """Convert a list of activity fractions to activities.

    Args:
        fractions: A numpy array with the activity fractions.
        one_dimensional: A bool that describes the shape of the
            input/output.

    Returns:
        A numpy array with the activities. If the array is
        multidimensional, each column corresponds to a subensemble
        from an expanded ensemble simulation.

    See Also:
        activities_to_fractions() for the opposite conversion.
    """
    if len(fractions.shape) == 1 or (one_dimensional and len(fractions) == 1):
        return np.exp(fractions)

    activities = np.copy(fractions)
    activity_sum = np.exp(fractions[0])
    activities[1:] *= activity_sum
    activities[0] = activity_sum - sum(activities[1:])

    return activities
