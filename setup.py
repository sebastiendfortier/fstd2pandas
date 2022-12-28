###############################################################################
# Copyright 2017-2021 - Climate Research Division
#                       Environment and Climate Change Canada
#
# This file is part of the "fstd2pandas" package.
#
# "fstd2pandas" is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# "fstd2pandas" is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with "fstd2pandas".  If not, see <http://www.gnu.org/licenses/>.
###############################################################################

from setuptools import setup, find_packages
with open("fstpy/VERSION",'r') as f:
  __version__ = f.readline().strip()

with open("README.md","r") as f:
  long_description = f.read()

setup (
  name="fstd2pandas",
  version=__version__,
  description = 'Converts RPN standard files (from Environment Canada) to pandas dataframes',
  long_description = long_description,
  long_description_content_type='text/markdown',
  url = 'https://github.com/sebastiendfortier/fstd2pandas',
  author="Sebastien Fortier",
  license = 'LGPL-3',
  classifiers = [
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering :: Atmospheric Science',
  ],
  packages = find_packages(),
  setup_requires = ['pip >= 8.1'],
  install_requires = ['pandas >= 1.2.4', 'numpy >= 1.19.5','fstd2nc', 'xarray >= 0.19.0', 'dask >= 2021.8.0', 'fstd2nc-deps >= 0.20200304.0'],
  package_data = {
    'fstpy': ['csv/*'],
  },
)
