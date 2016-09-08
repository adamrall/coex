"""Installation script for coex using setuptools."""

from setuptools import setup


setup(
    name='coex',
    version='1.0',
    packages=['coex'],
    install_requires=['numpy', 'scipy'],
    author='Adam Rall',
    author_email='arall@buffalo.edu',
    license='BSD',
    keywords='molecular simulation',
)
