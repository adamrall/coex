# setup.py
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

"""Installation script for coex using setuptools."""

from setuptools import setup


setup(
    name='coex',
    version='0.1',
    packages=['coex'],
    install_requires=['numpy', 'scipy'],
    author='Adam Rall',
    author_email='arall@buffalo.edu',
    license='BSD',
    keywords='molecular simulation',
)
