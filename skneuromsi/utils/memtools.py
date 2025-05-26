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
    """Dataclass representing memory impact.

    Parameters
    ----------
    expected_size : int
        The expected size of the memory impact in bytes.
    vmem : object
        The virtual memory object.

    """

    expected_size: int
    vmem: object

    @property
    def total_ratio(self):
        """Get the ratio of the expected size to the total memory."""
        return self.expected_size / self.vmem.total

    @property
    def available_ratio(self):
        """Get the ratio of the expected size to the available memory."""
        return self.expected_size / self.vmem.available

    @property
    def hexpected_size(self):
        """Get the human-readable representation of the expected size."""
        return humanize.naturalsize(self.expected_size)

    @property
    def havailable_memory(self):
        """Get the human-readable representation of the available memory."""
        return humanize.naturalsize(self.vmem.available)

    @property
    def htotal_memory(self):
        """Get the human-readable string representation of the total memory."""
        return humanize.naturalsize(self.vmem.total)

    def __repr__(self):
        """Return a string representation of the MemoryImpact object."""
        return (
            f"<memimpact expected_size={self.hexpected_size!r}, "
            f"total_ratio={self.total_ratio}, "
            f"available_ratio={self.available_ratio}>"
        )


def memory_impact(obj, *, size_factor=1, num_objects=1):
    """Calculate the memory impact of an object.

    Parameters
    ----------
    obj : object
        The object to calculate memory impact for.
    num_objects : int, optional
        Number of objects to consider (default is 1).
    size_factor : float, optional
        Factor to multiply the expected size by (default is 1).
        Size factor is a factor to multiply the expected size by.
        It is assumed that most objects are a little bigger or smaller
        than the expected size. For example, if `size_factor=1.2` then
        the expected size is multiplied by 1.2, this means that most
        objects will be 20% bigger than the expected size. If
        `size_factor=0.75` then most objects will be 35% smaller.

    Returns
    -------
    _MemoryImpact
        An instance of _MemoryImpact containing the calculated impact metrics.

    """
    if size_factor < 0:
        raise ValueError("size_factor must be a positive value")

    # the size of the object in the memory
    obj_memory = memory_usage(obj)

    # the expected size of the "num_objects" object in the memory
    # based on the size factor and the number of objects
    expected_size = num_objects * obj_memory.size * size_factor
    vmem = psutil.virtual_memory()

    return _MemoryImpact(
        expected_size=expected_size,
        vmem=vmem,
    )
