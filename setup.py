#!/usr/bin/env python3

import setuptools

setuptools.setup(
    name='edunets',
    version='1.0',
    description='A very basic deep learning framework built to better understand how neural nets work',
    license='MIT',
    packages=['edunets'],
    install_requires=['numpy', 'scipy', 'graphviz'],
    python_requires='>=3.6',
)