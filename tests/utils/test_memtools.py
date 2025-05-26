#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2025, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt


# =============================================================================
# DOCS
# =============================================================================

"""test for skneuromsi.utils.memtools"""

# =============================================================================
# IMPORTS
# =============================================================================


import humanize

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

    mi = memtools.memory_impact(obj, size_factor=1, num_objects=1)

    assert isinstance(mi.expected_size, int)
    assert isinstance(mi.total_ratio, float)
    assert isinstance(mi.available_ratio, float)
    assert isinstance(mi.hexpected_size, str)
    assert isinstance(mi.havailable_memory, str)
    assert isinstance(mi.htotal_memory, str)
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
