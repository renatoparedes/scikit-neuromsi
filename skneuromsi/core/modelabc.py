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

"""Implementation of a metaclass to create renameable parameters in functions \
and methods.

This module provides classes and utilities for creating renameable parameters
in functions and methods, as well as configuring the run method of
SKNMSIMethodABC subclasses.

"""

# =============================================================================
# IMPORTS
# =============================================================================

import functools
import inspect
import itertools as it
import re
import string
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from bidict import frozenbidict

from .ndresult import NDResult

# =============================================================================
# CONSTANTS
# =============================================================================

MODEL_TYPES = ("MLE", "Bayesian", "Neural")


# =============================================================================
#
# =============================================================================


@dataclass
class ParameterAliasTemplate:
    """Represents the rule for constructing aliases for a parameter.

    Parameters
    ----------
    target : str
        The name of the parameter to replace.
    template : string.Template
        The template used to generate aliases for the target.
    doc_pattern : re.Pattern, optional
        Regex pattern to replace all occurrences of the target with the alias
        in the documentation. If not provided, a default pattern is used.

    """

    target: str
    template: string.Template
    doc_pattern: str = None

    def __post_init__(self) -> None:
        """Initialize the object after it has been created.

        This method is called automatically after the object has been created
        and initializes the object's attributes. It checks if the `template`
        attribute is an instance of `string.Template` and if not,
        it converts it to one. It also checks if the `doc_pattern` attribute
        is an instance of `re.Pattern` and if not, it converts it to one.

        """
        if not isinstance(self.template, string.Template):
            self.template = string.Template(self.template)

        self.doc_pattern = (
            self.doc_pattern
            if self.doc_pattern
            else (r"(?<=\b)" + self.target + r"(?=\b)")
        )

        if not isinstance(self.doc_pattern, re.Pattern):
            self.doc_pattern = re.compile(self.doc_pattern)

    def __hash__(self):
        """Hash based on the target and template."""
        return hash((self.target, self.template))

    def __eq__(self, other):
        """Equality based on the target and template."""
        return (
            isinstance(other, type(self))
            and self.target == other.target
            and self.template.template == other.template.template
            and self.doc_pattern == other.doc_pattern
        )

    def __ne__(self, other):
        """Inequality based on the target and template."""
        return not self == other

    @property
    def template_variables(self) -> frozenset:
        """Variables del template."""
        tpl = self.template
        variables = set()
        for match in tpl.pattern.finditer(tpl.template):
            variables.add(match[match.lastgroup])
        return frozenset(variables)

    def render(self, context) -> str:
        """Crea un nuevo alias basado en el contexto provisto."""
        return self.template.substitute(context)


# =============================================================================
# CONFIG
# =============================================================================
@dataclass(kw_only=True)
class SKNMSIRunConfig:
    """Configuration class for using aliases in the run method and creating \
    the result object.

    Parameters
    ----------
    _input : tuple
        Input configuration for alias templates.
    _output : tuple
        Output configuration for alias templates.
    _model_name : str
        Name of the model.
    _model_type : str
        Type of the model.
    _output_mode : str
        Output mode of the model.

    """

    _input: tuple
    _output: tuple
    _model_name: str
    _model_type: str
    _output_mode: str

    # initialization

    def __post_init__(self):
        """Initialize the object after it has been created.

        This method is automatically called after the object has been created
        and initializes the `_input` and `_output` attributes as tuples. It
        also checks if the `_model_type` attribute is valid and raises a
        `ValueError` if it is not. Additionally, it checks for duplicate
        targets in the `_input` and `_output` attributes and raises a
        `ValueError` if any are found.

        """
        self._input = tuple(self._input)
        self._output = tuple(self._output)

        if self._model_type not in MODEL_TYPES:
            mtypes = ", ".join(MODEL_TYPES)
            raise ValueError(
                f"Class attribute '_model_type' must be one of {mtypes}"
            )

        for pats in [self._input, self._output]:
            targets = set()
            for pat in pats:
                if pat.target in targets:
                    target = pat.target
                    raise ValueError(
                        f"Duplicated ParameterAliasTemplate target {target}"
                    )
                targets.add(pat.target)

    @classmethod
    def from_method_class(cls, method_class):
        """Create a new configuration instance based on a subclass of \
        SKNMSIMethodABC.

        The class must implement the '_run_input' variable, which contains the
        configuration of the alias templates.

        Parameters
        ----------
        method_class : type
            Subclass of SKNMSIMethodABC.

        Returns
        -------
        SKNMSIRunConfig
            Configuration instance based on the method class.

        """

        def parse_run_conf(conf, name):
            cache = {}
            for patpl in conf:
                if isinstance(patpl, Mapping):
                    patpl = ParameterAliasTemplate(**patpl)

                elif isinstance(patpl, Iterable):
                    patpl = ParameterAliasTemplate(*patpl)

                if not isinstance(patpl, ParameterAliasTemplate):
                    raise TypeError(
                        f"All elements of '{name}' must be "
                        "instances of  'ParameterAliasTemplate' or parameters "
                        f"for their constructor. Found: {patpl!r}"
                    )
                cache[patpl.target] = patpl

            return tuple(cache.values())

        _input = parse_run_conf(method_class._run_input, "_run_input")
        _output = parse_run_conf(method_class._run_output, "_run_output")
        _model_name = str(
            getattr(method_class, "_model_name", method_class.__name__)
        )
        _model_type = method_class._model_type
        _output_mode = method_class._output_mode

        return cls(
            _model_name=_model_name,
            _model_type=_model_type,
            _input=_input,
            _output=_output,
            _output_mode=_output_mode,
        )

    # API

    @property
    def input_targets(self):
        """Set of all configurable parameters of the run method."""
        return frozenset(patpl.target for patpl in self._input)

    @property
    def output_targets(self):
        """Set of all configurable parameters of run method output."""
        return frozenset(patpl.target for patpl in self._output)

    @property
    def template_variables(self):
        """All alias-template variables defined in all targets."""
        template_variables = set()
        for patpl in it.chain(self._input, self._output):
            template_variables.update(patpl.template_variables)
        return frozenset(template_variables)

    def make_run_input_alias_map(self, context):
        """Create a bidirectional dictionary that maps aliases to targets.

        The context is the configuration provided by the user through the
        init parameters.

        Parameters
        ----------
        context : dict
            Context for rendering the aliases.

        Returns
        -------
        frozenbidict
            Bidirectional dictionary mapping aliases to targets.

        """
        ia_map = frozenbidict(
            (patpl.target, patpl.render(context)) for patpl in self._input
        )
        return ia_map

    def make_run_output_alias_map(self, context):
        """Create a bidirectional dictionary that maps output aliases to \
        targets.

        Parameters
        ----------
        context : dict
            Context for rendering the aliases.

        Returns
        -------
        frozenbidict
            Bidirectional dictionary mapping output aliases to targets.

        """
        oa_map = frozenbidict(
            (patpl.target, patpl.render(context)) for patpl in self._output
        )
        return oa_map

    # VALIDATION ==============================================================

    def _parameters_difference(self, method, expected_parameters):
        """Return the difference between the parameters of a method and a \
        set of expected values.

        Parameters
        ----------
        method : callable
            Method from which to extract the parameters.
        expected_parameters : iterable
            Collection of expected parameters that the method should have.

        Returns
        -------
        set
            All parameters that were expected but not found in the method.

        """
        parameters = inspect.signature(method).parameters
        parameters_difference = set(expected_parameters).difference(parameters)
        return parameters_difference

    def validate_init_and_run(self, method_class):
        """Validate that the __init__ and run methods have the appropriate \
        parameters.

        The method takes as argument a class inherited from SKNMSIMethodABC.

        Parameters
        ----------
        method_class : type
            Subclass of SKNMSIMethodABC.

        Raises
        ------
        TypeError
            If __init__ or run() do not have the required parameters.

        """
        # THE RUN ------------------------------------------------------------
        targets_not_found = self._parameters_difference(
            method_class.run, self.input_targets
        )
        if targets_not_found:
            tnf = ", ".join(targets_not_found)
            raise TypeError(
                f"Target/s {tnf!r} not found as parameter in 'run' method"
            )

        # THE INIT ------------------------------------------------------------
        tpl_not_found = self._parameters_difference(
            method_class.__init__, self.template_variables
        )
        if tpl_not_found:
            tnf = ", ".join(tpl_not_found)
            raise TypeError(
                f"Template variable/s {tnf!r} not found as "
                "parameter in '__init__' method"
            )

    # RUN METHOD REPLACEMENT ==================================================
    def _make_run_signature_with_alias(self, run_method, target_alias_map):
        """Create a new run method signature with aliased parameters.

        Parameters
        ----------
        run_method : callable
            Original run method.
        target_alias_map : dict
            Mapping of targets to aliases.

        Returns
        -------
        inspect.Signature
            New run method signature with aliased parameters.

        """
        # replace each "target" parameter with one with an alias
        # leaving unchanged the ones that are not targets
        # the rest of the parameter metadata remains the same
        run_signature = inspect.signature(run_method)
        aliased_run_parameters = []
        for run_parameter in run_signature.parameters.values():
            alias_parameter = inspect.Parameter(
                name=target_alias_map.get(
                    run_parameter.name, run_parameter.name
                ),
                kind=run_parameter.kind,
                default=run_parameter.default,
                annotation=run_parameter.annotation,
            )
            aliased_run_parameters.append(alias_parameter)

        signature_with_alias = run_signature.replace(
            parameters=aliased_run_parameters
        )

        return signature_with_alias

    def _make_run_doc_with_alias(self, run_method, target_alias_map):
        """Create a new run method docstring with aliased parameters.

        Parameters
        ----------
        run_method : callable
            Original run method.
        target_alias_map : dict
            Mapping of targets to aliases.

        Returns
        -------
        str
            New run method docstring with aliased parameters.

        """
        doc = run_method.__doc__ or ""
        for pat in self._input:
            pattern = pat.doc_pattern
            alias = target_alias_map.get(pat.target)
            doc = pattern.sub(alias, doc)
        return doc

    def wrap_run(self, model_instance, run_template_context):
        """Return a wrapper for the run method.

        This method is used *for each instance* of a class object
        ``SKNMSIMethodABC``.

        The resulting method accepts as parameters the resulting aliases from
        the configuration and maps them to the "targets".

        Parameters
        ----------
        model_instance : SKNMSIMethodABC
            Instance of the model class.
        run_template_context : dict
            Context for rendering the aliases.

        Returns
        -------
        callable
            Wrapped run method.

        """
        # extract the runmethod , calculate_causes
        # and calculate_perceived_positions
        run_method = model_instance.run
        calculate_causes_method = model_instance.calculate_causes

        # create a bidirectional mapping of alias and target in a dictionary
        # for the input and the output
        input_alias_map = self.make_run_input_alias_map(run_template_context)
        output_alias_map = self.make_run_output_alias_map(run_template_context)

        signature_with_alias = self._make_run_signature_with_alias(
            run_method, input_alias_map
        )

        doc_with_alias = self._make_run_doc_with_alias(
            run_method, input_alias_map
        )

        time_range = model_instance.time_range
        position_range = model_instance.position_range

        time_res = model_instance.time_res
        position_res = model_instance.position_res

        make_result = getattr(
            model_instance, "_make_result", NDResult.from_modes_dict
        )

        @functools.wraps(run_method)
        def run_wrapper(*args, **kwargs):
            # if some parameters are not valid in the aliased function
            # we must raise a type error with the correct message
            invalid_kws = set(kwargs).difference(
                signature_with_alias.parameters
            )

            if invalid_kws:
                invalid = invalid_kws.pop()
                raise TypeError(
                    f"run() got an unexpected keyword argument {invalid!r}"
                )

            # inside the wrapper, we bind the parameters
            # so we can assign the values to the appropriate parameter names
            bound_params = signature_with_alias.bind(*args, **kwargs)
            bound_params.apply_defaults()

            # now we separate the args from the kwargs and change the
            # aliases to the corresponding targets.
            target_args = bound_params.args
            target_kwargs = {
                input_alias_map.inv.get(k, k): v
                for k, v in bound_params.kwargs.items()
            }
            response, extra = run_method(*target_args, **target_kwargs)
            if self._output_mode not in response:
                raise ValueError(f"No output mode {self._output_mode} found")

            causes = calculate_causes_method(**response, **extra)

            # now we rename the output
            response_aliased = {
                output_alias_map.get(k, k): v for k, v in response.items()
            }
            extra_aliased = {
                output_alias_map.get(k, k): v for k, v in extra.items()
            }
            ndresult = make_result(
                mname=self._model_name,
                mtype=self._model_type,
                output_mode=self._output_mode,
                nmap=output_alias_map,
                modes_dict=response_aliased,
                time_range=time_range,
                position_range=position_range,
                time_res=time_res,
                position_res=position_res,
                causes=causes,
                run_parameters=dict(bound_params.arguments),
                extra=extra_aliased,
            )

            return ndresult

        run_wrapper.__skneuromsi_run_template_context__ = run_template_context
        run_wrapper.__signature__ = signature_with_alias
        run_wrapper.__doc__ = doc_with_alias

        return run_wrapper

    # WRAP INIT================================================================

    def wrap_init(self, init_method):
        """Wrap the __init__ method of an SKNMSIMethodABC subclass.

        Parameters
        ----------
        init_method : callable
            Original __init__ method.

        Returns
        -------
        callable
            Wrapped __init__ method.

        """
        signature = inspect.signature(init_method)

        @functools.wraps(init_method)
        def init_wrapper(instance, *args, **kwargs):
            # first, execute the old init and leave the object configured
            init_method(instance, *args, **kwargs)

            # bind all arguments to init, so we know exactly what value goes
            # with what parameter
            bound_args = signature.bind_partial(instance, *args, **kwargs)
            bound_args.apply_defaults()

            # grab the bound arguments and if that argument belongs to a
            # template variable, add it to the dictionary that will be used to
            # create aliases for run()
            run_template_context = {
                tvar: bound_args.arguments[tvar]
                for tvar in self.template_variables
            }

            # create the new run, and chant it at the instance level!
            instance.run = self.wrap_run(instance, run_template_context)

        return init_wrapper

    # MODEL GET-SET STATE =====================================================

    def get_model_state(self, instance):
        """Get the state of a model instance.

        Parameters
        ----------
        instance : SKNMSIMethodABC
            Model instance.

        Returns
        -------
        dict
            State of the model instance.

        """
        state = dict(instance.__dict__)

        # becase the run instance method is a clousure we can't serialize this
        # and we only store the context
        state["run"] = dict(instance.run.__skneuromsi_run_template_context__)

        return state

    def set_model_state(self, instance, state):
        """Set the state of a model instance.

        Parameters
        ----------
        instance : SKNMSIMethodABC
            Model instance.
        state : dict
            State to set on the model instance.

        """
        # remove the run_template_context to restore the instance runt
        run_template_context = state.pop("run")

        # set all the state except for the run
        instance.__dict__.update(state)

        # recreate the run with the correct context
        instance.run = self.wrap_run(instance, run_template_context)


# =============================================================================
# BASES
# =============================================================================

# THE REAL METHOD BASE ========================================================

# ALL SKNMSIMethodABC subclasses must redefine this attributes and methods
TO_REDEFINE = [
    #
    # class level
    ("_run_input", "Class attribute"),
    ("_run_output", "Class attribute"),
    ("_model_type", "Class attribute"),
    ("_output_mode", "Class attribute"),
    #
    # instance level redefinition
    ("time_range", "Attribute"),
    ("position_range", "Attribute"),
    ("time_res", "Attribute"),
    ("position_res", "Attribute"),
    ("run", "Method"),
    ("set_random", "Method"),
]


class SKNMSIMethodABC:
    """Abstract class that allows to configure method names dynamically.

    This class serves as the base class for all models in skneuromsi. It
    provides dynamic configuration of method names and parameters using
    aliases.

    """

    def __init_subclass__(cls):
        """Perform a validation and configuration of subclasses.

        This method is called when a new subclass of SKNMSIMethodABC is
        defined. It validates the subclass attributes and creates the
        SKNMSIRunConfig instance for the subclass.

        Raises
        ------
        TypeError
            If the subclass does not redefine the required attributes or
            methods.

        """
        # simple validation
        if vars(cls).get("_abstract", False):
            return

        # validate redefinitions
        for attr, attr_type in TO_REDEFINE:
            if not hasattr(cls, attr):
                cls_name = cls.__name__
                msg = (
                    f"{attr_type} '{attr}' must be redefined "
                    f"in class {cls_name}"
                )
                raise TypeError(msg)

        # config creation
        config = SKNMSIRunConfig.from_method_class(cls)

        # run and init method vs configuration validation
        config.validate_init_and_run(cls)

        # teardown
        for attr, _ in TO_REDEFINE:
            if attr.startswith("_"):
                delattr(cls, attr)

        cls.__init__ = config.wrap_init(cls.__init__)
        cls._run_config = config

    def __getstate__(self):
        """Get the state of a model instance.

        Needed for multiprocessing environment.

        """
        cls = type(self)
        return cls._run_config.get_model_state(self)

    def __setstate__(self, state):
        """Set the state of a model instance.

        Needed for multiprocessing environment.

        """
        cls = type(self)
        cls._run_config.set_model_state(self, state)

    def calculate_causes(self, **kwargs):
        """Calculate the causes based on the model output.

        This method should be overridden by subclasses to provide the specific
        implementation for calculating causes.

        Parameters
        ----------
        **kwargs
            Keyword arguments representing the model output.

        Returns
        -------
        Any
            The calculated causes.

        """
        return None
