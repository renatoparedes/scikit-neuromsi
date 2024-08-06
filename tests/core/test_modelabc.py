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

from skneuromsi.core import modelabc

# =============================================================================
# ParameterAliasTemplate
# =============================================================================


def test_ParameterAliasTemplate_init():
    pat = modelabc.ParameterAliasTemplate("foo", "$p0 ${p1}_foo")
    assert pat.target == "foo"
    assert isinstance(pat.template, string.Template)
    assert hash(pat) == hash((pat.target, pat.template))
    assert pat.template_variables == {"p0", "p1"}
    assert pat.render({"p0": "xxx", "p1": "yyy"}) == "xxx yyy_foo"

    assert pat == modelabc.ParameterAliasTemplate("foo", "$p0 ${p1}_foo")
    assert pat != modelabc.ParameterAliasTemplate("fuo", "${p0}_${p1}_foo")


# =============================================================================
# SKNMSIRunConfig
# =============================================================================


def test_SKNMSIRunConfig_init():
    pat0 = modelabc.ParameterAliasTemplate("foo", "${p0}_${p1}_foo")
    pat1 = modelabc.ParameterAliasTemplate("baz", "${p2}_${p3}_baz")

    pat2 = modelabc.ParameterAliasTemplate("foo", "${p0}_${p1}_foo")
    pat3 = modelabc.ParameterAliasTemplate("baz", "${p2}_${p4}_baz")

    config = modelabc.SKNMSIRunConfig(
        _input=[pat0, pat1],
        _output=[pat2, pat3],
        _result_cls=None,
        _model_name="model",
        _model_type="Neural",
        _output_mode="output_mode",
    )

    assert isinstance(config._input, tuple)
    assert isinstance(config._output, tuple)
    assert config._result_cls is None

    assert config._input == (pat0, pat1)
    assert config.input_targets == {"foo", "baz"}

    assert config._output == (pat2, pat3)
    assert config.output_targets == {"foo", "baz"}

    assert config.template_variables == {"p0", "p1", "p2", "p3", "p4"}

    input_alias_map = config.make_run_input_alias_map(
        {"p0": "w", "p1": "x", "p2": "y", "p3": "z"}
    )

    assert (
        input_alias_map.inv["w_x_foo"] == "foo"
        and input_alias_map.inv["y_z_baz"] == "baz"
    )
    assert (
        input_alias_map["foo"] == "w_x_foo"
        and input_alias_map["baz"] == "y_z_baz"
    )

    output_alias_map = config.make_run_output_alias_map(
        {"p0": "w", "p1": "x", "p2": "y", "p4": "z"}
    )
    assert (
        output_alias_map.inv["w_x_foo"] == "foo"
        and output_alias_map.inv["y_z_baz"] == "baz"
    )
    assert (
        output_alias_map["foo"] == "w_x_foo"
        and output_alias_map["baz"] == "y_z_baz"
    )

    assert config._output_mode == "output_mode"


def test_SKNMSIRunConfig_duplicated_targets():
    pat0 = modelabc.ParameterAliasTemplate("foo", "${p0}_${p1}_foo")
    pat1 = modelabc.ParameterAliasTemplate("foo", "${p2}_${p3}_baz")

    with pytest.raises(ValueError):
        modelabc.SKNMSIRunConfig(
            [pat0, pat1], [pat0], None, "model", "Neural", "output_mode"
        )

    with pytest.raises(ValueError):
        modelabc.SKNMSIRunConfig(
            [pat0], [pat0, pat1], None, "model", "Neural", "output_mode"
        )


def test_SKNMSIRunConfig_from_method_class():
    pat0 = modelabc.ParameterAliasTemplate("foo", "${p0}_${p1}_foo")
    pat1 = modelabc.ParameterAliasTemplate("baz", "${p2}_${p3}_baz")

    pat2 = modelabc.ParameterAliasTemplate("foo", "${p0}_${p1}_foo")
    pat3 = modelabc.ParameterAliasTemplate("baz", "${p2}_${p4}_baz")

    class MethodClass:
        _run_input = [pat0, pat1]
        _run_output = [pat2, pat3]
        _run_result = None
        _model_type = "Neural"
        _output_mode = "output_mode"

    config = modelabc.SKNMSIRunConfig.from_method_class(MethodClass)

    assert isinstance(config._input, tuple)
    assert isinstance(config._output, tuple)
    assert config._result_cls is None

    assert config._input == (pat0, pat1)
    assert config.input_targets == {"foo", "baz"}

    assert config._output == (pat2, pat3)
    assert config.output_targets == {"foo", "baz"}

    assert config.template_variables == {"p0", "p1", "p2", "p3", "p4"}

    input_alias_map = config.make_run_input_alias_map(
        {"p0": "w", "p1": "x", "p2": "y", "p3": "z"}
    )

    assert (
        input_alias_map.inv["w_x_foo"] == "foo"
        and input_alias_map.inv["y_z_baz"] == "baz"
    )
    assert (
        input_alias_map["foo"] == "w_x_foo"
        and input_alias_map["baz"] == "y_z_baz"
    )

    output_alias_map = config.make_run_output_alias_map(
        {"p0": "w", "p1": "x", "p2": "y", "p4": "z"}
    )
    assert (
        output_alias_map.inv["w_x_foo"] == "foo"
        and output_alias_map.inv["y_z_baz"] == "baz"
    )
    assert (
        output_alias_map["foo"] == "w_x_foo"
        and output_alias_map["baz"] == "y_z_baz"
    )


def test_SKNMSIRunConfig_from_method_class_with_tuples():
    pat0 = ("foo", "${p0}_${p1}_foo")
    pat1 = ("baz", "${p2}_${p3}_baz")

    pat2 = ("foo", "${p0}_${p1}_foo")
    pat3 = ("baz", "${p2}_${p4}_baz")

    class MethodClass:
        _run_input = [pat0, pat1]
        _run_output = [pat2, pat3]
        _run_result = None
        _model_type = "Neural"
        _output_mode = "output_mode"

    config = modelabc.SKNMSIRunConfig.from_method_class(MethodClass)

    assert isinstance(config._input, tuple)
    assert isinstance(config._output, tuple)
    assert config._result_cls is None

    assert config._input == (
        modelabc.ParameterAliasTemplate(*pat0),
        modelabc.ParameterAliasTemplate(*pat1),
    )
    assert config.input_targets == {"foo", "baz"}

    assert config._output == (
        modelabc.ParameterAliasTemplate(*pat2),
        modelabc.ParameterAliasTemplate(*pat3),
    )
    assert config.output_targets == {"foo", "baz"}

    assert config.template_variables == {"p0", "p1", "p2", "p3", "p4"}

    input_alias_map = config.make_run_input_alias_map(
        {"p0": "w", "p1": "x", "p2": "y", "p3": "z"}
    )

    assert (
        input_alias_map.inv["w_x_foo"] == "foo"
        and input_alias_map.inv["y_z_baz"] == "baz"
    )
    assert (
        input_alias_map["foo"] == "w_x_foo"
        and input_alias_map["baz"] == "y_z_baz"
    )

    output_alias_map = config.make_run_output_alias_map(
        {"p0": "w", "p1": "x", "p2": "y", "p4": "z"}
    )
    assert (
        output_alias_map.inv["w_x_foo"] == "foo"
        and output_alias_map.inv["y_z_baz"] == "baz"
    )
    assert (
        output_alias_map["foo"] == "w_x_foo"
        and output_alias_map["baz"] == "y_z_baz"
    )


def test_SKNMSIRunConfig_from_method_class_with_dicts():
    pat0 = {"target": "foo", "template": "${p0}_${p1}_foo"}
    pat1 = {"target": "baz", "template": "${p2}_${p3}_baz"}

    pat2 = {"target": "foo", "template": "${p0}_${p1}_foo"}
    pat3 = {"target": "baz", "template": "${p2}_${p4}_baz"}

    class MethodClass:
        _run_input = [pat0, pat1]
        _run_output = [pat2, pat3]
        _run_result = None
        _model_type = "Neural"
        _output_mode = "output_mode"

    config = modelabc.SKNMSIRunConfig.from_method_class(MethodClass)

    assert isinstance(config._input, tuple)
    assert isinstance(config._output, tuple)
    assert config._result_cls is None

    assert config._input == (
        modelabc.ParameterAliasTemplate(**pat0),
        modelabc.ParameterAliasTemplate(**pat1),
    )
    assert config.input_targets == {"foo", "baz"}

    assert config._output == (
        modelabc.ParameterAliasTemplate(**pat2),
        modelabc.ParameterAliasTemplate(**pat3),
    )
    assert config.output_targets == {"foo", "baz"}

    assert config.template_variables == {"p0", "p1", "p2", "p3", "p4"}

    input_alias_map = config.make_run_input_alias_map(
        {"p0": "w", "p1": "x", "p2": "y", "p3": "z"}
    )

    assert (
        input_alias_map.inv["w_x_foo"] == "foo"
        and input_alias_map.inv["y_z_baz"] == "baz"
    )
    assert (
        input_alias_map["foo"] == "w_x_foo"
        and input_alias_map["baz"] == "y_z_baz"
    )

    output_alias_map = config.make_run_output_alias_map(
        {"p0": "w", "p1": "x", "p2": "y", "p4": "z"}
    )
    assert (
        output_alias_map.inv["w_x_foo"] == "foo"
        and output_alias_map.inv["y_z_baz"] == "baz"
    )
    assert (
        output_alias_map["foo"] == "w_x_foo"
        and output_alias_map["baz"] == "y_z_baz"
    )


def test_SKNMSIRunConfig_from_method_class_invalid_type():
    class MethodClass:
        _run_input = [None]

    with pytest.raises(TypeError):
        modelabc.SKNMSIRunConfig.from_method_class(MethodClass)


def test_SKNMSIRunConfig_validate_init_and_run():
    pat0 = modelabc.ParameterAliasTemplate("foo", "${p0}_${p1}_foo")
    pat1 = modelabc.ParameterAliasTemplate("baz", "${p2}_${p3}_baz")

    config = modelabc.SKNMSIRunConfig(
        [pat0, pat1],
        [pat0, pat1],
        None,
        _model_name="model",
        _model_type="Neural",
        _output_mode="output_mode",
    )

    class MethodClass:
        def __init__(self, p0, p1, p2, p3, p4):
            pass

        def run(self, foo, baz):
            pass

    assert config.validate_init_and_run(MethodClass) is None


def test_SKNMSIRunConfig_validate_init_and_run_missing_target():
    pat0 = modelabc.ParameterAliasTemplate("foo", "${p0}_${p1}_foo")
    pat1 = modelabc.ParameterAliasTemplate("baz", "${p2}_${p3}_baz")

    config = modelabc.SKNMSIRunConfig(
        [pat0, pat1],
        [pat0, pat1],
        None,
        _model_name="model",
        _model_type="Neural",
        _output_mode="output_mode",
    )

    class MethodClass:
        def __init__(self, p0, p1, p2, p3, p4):
            pass

        def run(self, baz):
            pass

    with pytest.raises(TypeError) as err:
        config.validate_init_and_run(MethodClass)

    err.match("Target/s 'foo' not found as parameter in 'run' method")


def test_SKNMSIRunConfig_validate_init_and_run_missing_template_variable():
    pat0 = modelabc.ParameterAliasTemplate("foo", "${p0}_${p1}_foo")
    pat1 = modelabc.ParameterAliasTemplate("baz", "${p2}_${p3}_baz")

    config = modelabc.SKNMSIRunConfig(
        [pat0, pat1],
        [pat0, pat1],
        None,
        _model_name="model",
        _model_type="Neural",
        _output_mode="output_mode",
    )

    class MethodClass:
        def __init__(self, p1, p2, p3, p4):
            pass

        def run(self, foo, baz):
            pass

    with pytest.raises(TypeError) as err:
        config.validate_init_and_run(MethodClass)

    err.match("Template variable/s 'p0' not found as parameter in '__init__'")


def test_SKNMSIRunConfig_invalid_model_type():
    with pytest.raises(ValueError):
        modelabc.SKNMSIRunConfig(
            [modelabc.ParameterAliasTemplate("foo", "${p0}_foo")],
            [],
            lambda **kw: {},
            "model",
            _model_type="Foo",
            _output_mode="output_mode",
        )


def test_SKNMSIRunConfig_wrap_run():
    config = modelabc.SKNMSIRunConfig(
        [modelabc.ParameterAliasTemplate("foo", "${p0}_foo")],
        [],
        lambda **kw: {},
        "model",
        _model_type="Neural",
        _output_mode="output_mode",
    )

    foo_calls = []

    class MethodClass:
        time_range = (1, 2)
        position_range = (1, 2)
        time_res = 1
        position_res = 2

        def run(self, foo):
            """foo: zaraza foo"""
            foo_calls.append({"foo": foo})
            return {"output_mode": [1, 2]}, {}

        def calculate_causes(self, **kwargs):
            return None

    new_run = config.wrap_run(MethodClass(), {"p0": "x"})

    assert new_run.__doc__ == "x_foo: zaraza x_foo"

    new_run(x_foo="zaraza")

    assert foo_calls == [{"foo": "zaraza"}]

    with pytest.raises(TypeError) as err:
        new_run()

    err.match("missing a required argument: 'x_foo'")

    with pytest.raises(TypeError) as err:
        new_run(foo="zaraza")

    err.match(r"run\(\) got an unexpected keyword argument 'foo'")


def test_SKNMSIRunConfig_wrap_init():
    config = modelabc.SKNMSIRunConfig(
        _input=[modelabc.ParameterAliasTemplate("foo", "${p0}_foo")],
        _output=[],
        _result_cls=lambda **kw: {},
        _model_name="model",
        _model_type="Neural",
        _output_mode="output_mode",
    )

    foo_calls = []

    class MethodClass:
        time_range = (1, 2)
        position_range = (1, 2)
        time_res = 1
        position_res = 2

        def __init__(self, p0):
            pass

        def run(self, foo):
            foo_calls.append({"self": self, "foo": foo})
            return {"output_mode": [1, 2]}, {}

        def calculate_causes(self, **kwargs):
            return None

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

    class Method(modelabc.SKNMSIMethodABC):
        _model_type = "Neural"
        _run_input = [
            {"target": "foo", "template": "${p0}_foo"},
        ]
        _run_output = [
            {"target": "foo", "template": "${p0}_foo"},
        ]
        time_range = (1, 2)
        position_range = (1, 2)
        time_res = 1
        position_res = 2

        def __init__(self, p0):
            pass

        def run(self, foo):
            foo_calls.append({"self": self, "foo": foo})
            return {}, {}

        def set_random(self):
            pass

    assert isinstance(Method._run_config, modelabc.SKNMSIRunConfig)

    instance = Method(p0="x")
    instance.run(x_foo="zaraza")

    assert foo_calls == [{"self": instance, "foo": "zaraza"}]

    with pytest.raises(TypeError) as err:
        instance.run()

    err.match("missing a required argument: 'x_foo'")

    with pytest.raises(TypeError) as err:
        instance.run(foo="zaraza")

    err.match(r"run\(\) got an unexpected keyword argument 'foo'")


@pytest.mark.parametrize("ignore, err_msg", modelabc.TO_REDEFINE)
def test_SKNMSIMethodABC_something_is_not_redefined(ignore, err_msg):
    content = {
        aname: object() for aname, _ in modelabc.TO_REDEFINE if aname != ignore
    }

    with pytest.raises(TypeError) as err:
        type("Foo", (modelabc.SKNMSIMethodABC,), content)

    err.match(err_msg)


def test_SKNMSIMethodABC_abstract():
    class Method(modelabc.SKNMSIMethodABC):
        _abstract = True
        _run_input = [
            {"target": "foo", "template": "${p0}_foo"},
        ]

    assert Method._run_input == [{"target": "foo", "template": "${p0}_foo"}]
