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


"""Tests for the `skneuromsi.utils.doctools` module."""


# =============================================================================
# IMPORTS
# =============================================================================

import warnings

import pytest

from skneuromsi.utils.doctools import doc_inherit


# =============================================================================
# TESTS
# =============================================================================


def test_doc_inherit_function():
    """Test the `doc_inherit` decorator on a function."""

    def parent_func():
        """Parent function docstring."""
        pass

    @doc_inherit(parent_func)
    def child_func():
        pass

    assert child_func.__doc__ == "Parent function docstring."


def test_doc_inherit_class():
    """Test the `doc_inherit` decorator on a class."""

    class ParentClass:
        """Parent class docstring."""

        def method(self):
            """Parent method docstring."""
            pass

    @doc_inherit(ParentClass, warn_class=False)
    class ChildClass(ParentClass):
        def method(self):
            pass

    assert ChildClass.__doc__ == "Parent class docstring."
    assert ChildClass.method.__doc__ != "Parent method docstring."


def test_doc_inherit_class_warning():
    """Test the `doc_inherit` decorator on a class with warning."""

    class ParentClass:
        """Parent class docstring."""

    with warnings.catch_warnings(record=True) as w:

        @doc_inherit(ParentClass)
        class ChildClass(ParentClass):
            pass

        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
