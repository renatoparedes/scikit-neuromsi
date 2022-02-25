#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2022, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

# =============================================================================
# IMPORTS
# =============================================================================

import string

import pytest

from skneuromsi import core

# =============================================================================
# ParameterAliasTemplate
# =============================================================================


def test_ParameterAliasTemplate_init():
    pat = core.ParameterAliasTemplate("foo", "$p0 ${p1}_foo")
    assert pat.target == "foo"
    assert isinstance(pat.template, string.Template)
    assert hash(pat) == hash((pat.target, pat.template))
    assert pat.template_variables == {"p0", "p1"}
    assert pat.render({"p0": "xxx", "p1": "yyy"}) == "xxx yyy_foo"

    assert pat == core.ParameterAliasTemplate("foo", "$p0 ${p1}_foo")
    assert pat != core.ParameterAliasTemplate("fuo", "${p0}_${p1}_foo")


# =============================================================================
# SKNMSIRunConfig
# =============================================================================


def test_SKNMSIRunConfig_init():
    pat0 = core.ParameterAliasTemplate("foo", "${p0}_${p1}_foo")
    pat1 = core.ParameterAliasTemplate("baz", "${p2}_${p3}_baz")

    config = core.SKNMSIRunConfig([pat0, pat1])

    assert isinstance(config.parameter_alias_templates, tuple)
    assert config.parameter_alias_templates == (pat0, pat1)
    assert config.targets == {"foo", "baz"}
    assert config.template_variables == {"p0", "p1", "p2", "p3"}

    alias_map = config.create_alias_target_map(
        {"p0": "w", "p1": "x", "p2": "y", "p3": "z"}
    )

    assert alias_map["w_x_foo"] == "foo" and alias_map["y_z_baz"] == "baz"
    assert (
        alias_map.inv["foo"] == "w_x_foo" and alias_map.inv["baz"] == "y_z_baz"
    )


def test_SKNMSIRunConfig_duplicated_targets():
    pat0 = core.ParameterAliasTemplate("foo", "${p0}_${p1}_foo")
    pat1 = core.ParameterAliasTemplate("foo", "${p2}_${p3}_baz")

    with pytest.raises(ValueError):
        core.SKNMSIRunConfig([pat0, pat1])


def test_SKNMSIRunConfig_from_method_class():
    pat0 = core.ParameterAliasTemplate("foo", "${p0}_${p1}_foo")
    pat1 = core.ParameterAliasTemplate("baz", "${p2}_${p3}_baz")

    class MethodClass:
        _sknms_run_method_config = [pat0, pat1]

    config = core.SKNMSIRunConfig.from_method_class(MethodClass)

    assert isinstance(config.parameter_alias_templates, tuple)
    assert config.parameter_alias_templates == (pat0, pat1)
    assert config.targets == {"foo", "baz"}
    assert config.template_variables == {"p0", "p1", "p2", "p3"}

    alias_map = config.create_alias_target_map(
        {"p0": "w", "p1": "x", "p2": "y", "p3": "z"}
    )

    assert alias_map["w_x_foo"] == "foo" and alias_map["y_z_baz"] == "baz"
    assert (
        alias_map.inv["foo"] == "w_x_foo" and alias_map.inv["baz"] == "y_z_baz"
    )


def test_SKNMSIRunConfig_from_method_class_with_tuples():
    pat0 = ("foo", "${p0}_${p1}_foo")
    pat1 = ("baz", "${p2}_${p3}_baz")

    class MethodClass:
        _sknms_run_method_config = [pat0, pat1]

    config = core.SKNMSIRunConfig.from_method_class(MethodClass)

    assert isinstance(config.parameter_alias_templates, tuple)
    assert config.parameter_alias_templates == (
        core.ParameterAliasTemplate(*pat0),
        core.ParameterAliasTemplate(*pat1),
    )
    assert config.targets == {"foo", "baz"}
    assert config.template_variables == {"p0", "p1", "p2", "p3"}

    alias_map = config.create_alias_target_map(
        {"p0": "w", "p1": "x", "p2": "y", "p3": "z"}
    )

    assert alias_map["w_x_foo"] == "foo" and alias_map["y_z_baz"] == "baz"
    assert (
        alias_map.inv["foo"] == "w_x_foo" and alias_map.inv["baz"] == "y_z_baz"
    )


def test_SKNMSIRunConfig_from_method_class_with_dicts():
    pat0 = {"target": "foo", "template": "${p0}_${p1}_foo"}
    pat1 = {"target": "baz", "template": "${p2}_${p3}_baz"}

    class MethodClass:
        _sknms_run_method_config = [pat0, pat1]

    config = core.SKNMSIRunConfig.from_method_class(MethodClass)

    assert isinstance(config.parameter_alias_templates, tuple)
    assert config.parameter_alias_templates == (
        core.ParameterAliasTemplate(**pat0),
        core.ParameterAliasTemplate(**pat1),
    )
    assert config.targets == {"foo", "baz"}
    assert config.template_variables == {"p0", "p1", "p2", "p3"}

    alias_map = config.create_alias_target_map(
        {"p0": "w", "p1": "x", "p2": "y", "p3": "z"}
    )

    assert alias_map["w_x_foo"] == "foo" and alias_map["y_z_baz"] == "baz"
    assert (
        alias_map.inv["foo"] == "w_x_foo" and alias_map.inv["baz"] == "y_z_baz"
    )


def test_SKNMSIRunConfig_from_method_class_invalid_type():
    class MethodClass:
        _sknms_run_method_config = [None]

    with pytest.raises(TypeError):
        core.SKNMSIRunConfig.from_method_class(MethodClass)


def test_SKNMSIRunConfig_validate_init_and_run():
    pat0 = core.ParameterAliasTemplate("foo", "${p0}_${p1}_foo")
    pat1 = core.ParameterAliasTemplate("baz", "${p2}_${p3}_baz")

    config = core.SKNMSIRunConfig([pat0, pat1])

    class MethodClass:
        def __init__(self, p0, p1, p2, p3, p4):
            pass

        def run(self, foo, baz):
            pass

    assert config.validate_init_and_run(MethodClass) is None


def test_SKNMSIRunConfig_validate_init_and_run_missing_target():
    pat0 = core.ParameterAliasTemplate("foo", "${p0}_${p1}_foo")
    pat1 = core.ParameterAliasTemplate("baz", "${p2}_${p3}_baz")

    config = core.SKNMSIRunConfig([pat0, pat1])

    class MethodClass:
        def __init__(self, p0, p1, p2, p3, p4):
            pass

        def run(self, baz):
            pass

    with pytest.raises(TypeError) as err:
        config.validate_init_and_run(MethodClass)

    err.match("Target/s 'foo' not found as parameter in 'run' method")


def test_SKNMSIRunConfig_validate_init_and_run_missing_template_variable():
    pat0 = core.ParameterAliasTemplate("foo", "${p0}_${p1}_foo")
    pat1 = core.ParameterAliasTemplate("baz", "${p2}_${p3}_baz")

    config = core.SKNMSIRunConfig([pat0, pat1])

    class MethodClass:
        def __init__(self, p1, p2, p3, p4):
            pass

        def run(self, foo, baz):
            pass

    with pytest.raises(TypeError) as err:
        config.validate_init_and_run(MethodClass)

    err.match("Template variable/s 'p0' not found as parameter in '__init__'")


def test_wrap_run():

    config = core.SKNMSIRunConfig(
        [core.ParameterAliasTemplate("foo", "${p0}_foo")]
    )

    foo_calls = []

    class MethodClass:
        def run(self, foo):
            foo_calls.append({"self": self, "foo": foo})

    new_run = config.wrap_run(MethodClass.run, {"p0": "x"})
    new_run(self=None, x_foo="zaraza")

    assert foo_calls == [{"self": None, "foo": "zaraza"}]

    with pytest.raises(TypeError) as err:
        new_run(self=None)

    err.match("missing a required argument: 'x_foo'")

    with pytest.raises(TypeError) as err:
        new_run(foo="zaraza")

    err.match(r"run\(\) got an unexpected keyword argument 'foo'")


def test_wrap_init():

    config = core.SKNMSIRunConfig(
        [core.ParameterAliasTemplate("foo", "${p0}_foo")]
    )

    foo_calls = []

    class MethodClass:
        def __init__(self, p0):
            pass

        def run(self, foo):
            foo_calls.append({"self": self, "foo": foo})

    new_init = config.wrap_init(MethodClass.__init__)

    instance = MethodClass(None)

    instance.run(foo="zaraza")
    assert foo_calls == [{"self": instance, "foo": "zaraza"}]

    new_init(instance, p0="x")

    instance.run(x_foo="fafafa")

    assert foo_calls == [
        {"self": instance, "foo": "zaraza"},
        {"self": instance, "foo": "fafafa"},
    ]

    with pytest.raises(TypeError) as err:
        instance.run()

    err.match("missing a required argument: 'x_foo'")

    with pytest.raises(TypeError) as err:
        instance.run(foo="zaraza")

    err.match(r"run\(\) got an unexpected keyword argument 'foo'")


# =============================================================================
# SKNMSIMethodABC
# =============================================================================


def test_SKNMSIMethodABC():

    foo_calls = []

    class Method(core.SKNMSIMethodABC):

        _sknms_run_method_config = [
            {"target": "foo", "template": "${p0}_foo"},
        ]

        def __init__(self, p0):
            pass

        def run(self, foo):
            foo_calls.append({"self": self, "foo": foo})

    assert isinstance(Method._sknms_run_method_config, core.SKNMSIRunConfig)

    instance = Method(p0="x")
    instance.run(x_foo="zaraza")

    assert foo_calls == [{"self": instance, "foo": "zaraza"}]

    with pytest.raises(TypeError) as err:
        instance.run()

    err.match("missing a required argument: 'x_foo'")

    with pytest.raises(TypeError) as err:
        instance.run(foo="zaraza")

    err.match(r"run\(\) got an unexpected keyword argument 'foo'")


def test_SKNMSIMethodABC_abstract():
    class Method(core.SKNMSIMethodABC):

        _sknms_abstract = True
        _sknms_run_method_config = [
            {"target": "foo", "template": "${p0}_foo"},
        ]

    assert Method._sknms_run_method_config == [
        {"target": "foo", "template": "${p0}_foo"}
    ]


def test_SKNMSIMethodABC_missing_run():
    with pytest.raises(TypeError) as err:

        class Method(core.SKNMSIMethodABC):

            _sknms_run_method_config = [
                {"target": "foo", "template": "${p0}_foo"},
            ]

    err.match("'run' method must be redefined")


def test_SKNMSIMethodABC_missing__sknms_run_method_config():
    with pytest.raises(TypeError) as err:

        class Method(core.SKNMSIMethodABC):
            def run(self):
                pass

    err.match("Class attribute '_sknms_run_method_config' must be redefined")


def test_SKNMSIMethodABC_base_run_not_implemethed():
    class Method(core.SKNMSIMethodABC):

        _sknms_run_method_config = []

        def __init__(self):
            pass

        def run(self):
            return super().run()

    instance = Method()

    with pytest.raises(NotImplementedError) as err:
        instance.run()

    err.match("Default run method has has no implentation")
