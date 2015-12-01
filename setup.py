"""Installation script for coex using setuptools."""

from setuptools import setup, find_packages


setup(
    name='coex',
    version='0.1',
    packages=['coex'],
    install_requires=['numpy', 'scipy'],
    author='Adam Rall',
    author_email='arall@buffalo.edu',
    license='MIT',
    keywords='molecular simulation',
)
