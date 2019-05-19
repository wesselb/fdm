# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from setuptools import find_packages, setup

with open('README.md') as f:
    readme = f.read()

with open('LICENCE') as f:
    licence = f.read()

setup(
    name='fdm',
    version='0.1.0',
    description='Compute derivatives with finite-difference methods',
    long_description=readme,
    author='Wessel Bruinsma',
    author_email='wessel.p.bruinsma@gmail.com',
    url='https://github.com/wesselb/fdm',
    license=licence,
    packages=find_packages(exclude=('tests', 'docs'))
)