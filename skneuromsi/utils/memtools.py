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

"""
Memory usage and impact analysis tools.

This module provides classes and functions to calculate and represent memory
usage and impact of Python objects.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import dataclasses as dclss

import humanize

import psutil

from pympler import asizeof


# =============================================================================
# MEMORY USAGE
# =============================================================================


@dclss.dataclass(frozen=True)
class _MemoryUsage:
    """
    Dataclass representing memory usage.

    Parameters
    ----------
    size : int
        The size of the memory usage in bytes.

    Attributes
    ----------
    size : int
        The size of the memory usage in bytes.
    hsize : str
        The human-readable string representation of the memory usage size.
    """

    size: int

    @property
    def hsize(self):
        """
        Get the human-readable string representation of the memory usage size.

        Returns
        -------
        str
            Human-readable string representation of the memory usage size.
        """
        return humanize.naturalsize(self.size)

    def __repr__(self):
        return f"<memusage {self.hsize!r}>"


def memory_usage(obj):
    """
    Calculate the memory usage of an object.

    Parameters
    ----------
    obj : object
        The object to calculate memory usage for.

    Returns
    -------
    _MemoryUsage
        An instance of _MemoryUsage containing the calculated memory size.
    """
    size = asizeof.asizeof(obj)
    return _MemoryUsage(size=size)


# =============================================================================
# MEMORY IMPACT
# =============================================================================
@dclss.dataclass(frozen=True, slots=True)
class _MemoryImpact:
    """
    Dataclass representing memory impact.

    Parameters
    ----------
    total_ratio : float
        Ratio of object memory to total system memory.
    available_ratio : float
        Ratio of object memory to available system memory.
    expected_size : int
        Expected size of the object(s) in bytes.

    Attributes
    ----------
    total_ratio : float
        Ratio of object memory to total system memory.
    available_ratio : float
        Ratio of object memory to available system memory.
    expected_size : int
        Expected size of the object(s) in bytes.
    hexpected_size : str
        Human-readable string representation of the expected size.
    """

    total_ratio: float
    available_ratio: float
    expected_size: int
    vmem: object

    @property
    def hexpected_size(self):
        """
        Get the human-readable string representation of the expected size.

        Returns
        -------
        str
            Human-readable string representation of the expected size.
        """
        return humanize.naturalsize(self.expected_size)

    @property
    def havailable_memory(self):
        """
        Get the human-readable string representation of the available memory.

        Returns
        -------
        str
            Human-readable string representation of the available memory.
        """
        return humanize.naturalsize(self.vmem.available)

    @property
    def htotal_memory(self):
        """
        Get the human-readable string representation of the total memory.

        Returns
        -------
        str
            Human-readable string representation of the total memory.
        """
        return humanize.naturalsize(self.vmem.total)

    def __repr__(self):
        return (
            f"<memimpact expected_size={self.hexpected_size!r}, "
            f"total_ratio={self.total_ratio}, "
            f"available_ratio={self.available_ratio}>"
        )


def memory_impact(obj, num_objects=1):
    """
    Calculate the memory impact of an object or multiple objects.

    Parameters
    ----------
    obj : object
        The object to calculate memory impact for.
    num_objects : int, optional
        Number of objects to consider (default is 1).

    Returns
    -------
    _MemoryImpact
        An instance of _MemoryImpact containing the calculated impact metrics.
    """
    obj_memory = memory_usage(obj)
    total_object_memory = obj_memory.size * num_objects
    vmem = psutil.virtual_memory()

    total_ratio = total_object_memory / vmem.total
    available_ratio = total_object_memory / vmem.available

    return _MemoryImpact(
        total_ratio=total_ratio,
        available_ratio=available_ratio,
        expected_size=total_object_memory,
        vmem=vmem,
    )
