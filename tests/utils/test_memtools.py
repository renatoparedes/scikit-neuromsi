#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2022, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt


# =============================================================================
# DOCS
# =============================================================================

"""test for skneuromsi.utils.memtools

"""

# =============================================================================
# IMPORTS
# =============================================================================


import humanize

import psutil

from pympler import asizeof

import pytest

from skneuromsi.utils import memtools

# =============================================================================
# TESTS MEM USAGE
# =============================================================================


def test_memory_usage():
    obj = 1
    expected_size = asizeof.asizeof(obj)
    expected_humanize = humanize.naturalsize(expected_size)

    memory_usage = memtools.memory_usage(obj)
    assert memory_usage.size == expected_size
    assert memory_usage.hsize == expected_humanize
    assert repr(memory_usage) == f"<memusage {expected_humanize!r}>"


def test_memory_impact():
    """Test the `memory_impact` function."""
    obj = 1
    expected_size = asizeof.asizeof(obj)
    expected_humanize = humanize.naturalsize(expected_size)
    vmem = psutil.virtual_memory()

    mi = memtools.memory_impact(obj, size_factor=1, num_objects=1)
    assert mi.expected_size == expected_size
    assert mi.vmem == vmem
    assert mi.total_ratio == expected_size / vmem.total
    assert mi.available_ratio == expected_size / vmem.available
    assert mi.hexpected_size == expected_humanize
    assert mi.havailable_memory == humanize.naturalsize(vmem.available)
    assert mi.htotal_memory == humanize.naturalsize(vmem.total)
    assert repr(mi) == (
        f"<memimpact expected_size={expected_humanize!r}, "
        f"total_ratio={mi.total_ratio}, "
        f"available_ratio={mi.available_ratio}>"
    )


def test_memory_impact_size_factor_lt_0():
    """Test the `memory_impact` function with size_factor < 0."""
    obj = 1
    with pytest.raises(ValueError):
        memtools.memory_impact(obj, size_factor=-1, num_objects=1)
