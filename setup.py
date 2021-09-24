#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-Neuromsi Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021, Renato Paredes
# License: ...
#   Full Text: ....

# =====================================================================
# DOCS
# =====================================================================

"""This file is for distribute and install Scikit-Neuromsi"""

# ======================================================================
# IMPORTS
# ======================================================================

import os
import pathlib

from setuptools import setup  # noqa

# =============================================================================
# CONSTANTS
# =============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))


REQUIREMENTS = ["numpy", "pandas", "attrs", "matplotlib"]

with open(PATH / "skneuromsi" / "__init__.py") as fp:
    for line in fp.readlines():
        if line.startswith("__version__ = "):
            VERSION = line.split("=", 1)[-1].replace('"', "").strip()
            break


with open("README.md") as fp:
    LONG_DESCRIPTION = fp.read()


# =============================================================================
# FUNCTIONS
# =============================================================================

setup(
    name="scikit-neuromsi",
    version=VERSION,
    description="Implementation of multisensory integration models in Python",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Renato Paredes",
    author_email="paredesrenato92@gmail.com",
    url="https://github.com/renatoparedes/scikit-neuromsi",
    packages=[
        "skneuromsi",
    ],
    license="The MIT License",
    install_requires=REQUIREMENTS,
    keywords=[
        "multisensory integration",
        "computational neuroscience",
        "cognitive modelling",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
    ],
    include_package_data=True,
)
