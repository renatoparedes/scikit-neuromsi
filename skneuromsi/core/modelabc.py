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

import functools
import inspect
import itertools as it
import re
import string
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import methodtools

from bidict import frozenbidict

import numpy as np

from . import result


# =============================================================================
# DOCS
# =============================================================================

"""Implementation of a metaclass to create renameable parameters in functions \
and methods.

"""

# =============================================================================
# CONSTANTS
# =============================================================================

MODEL_TYPES = ("MLE", "Bayesian", "Neural")


# =============================================================================
#
# =============================================================================


@dataclass
class ParameterAliasTemplate:
    """Representa la regla de como construir alias para algun parámetro.

    Parameters
    ----------
    target: str
        Es el nombre del parametro a reemplazar.
    template: string.Template
        Es el template sobre el cual se generaran los alias para el target.
    doc_pattern: re.Pattern
        Regex para reemplazar todas las ocurrencias del target por el alias
        en la documentación.

    """

    target: str
    template: string.Template
    doc_pattern: str = None

    def __post_init__(self) -> None:
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
        return hash((self.target, self.template))

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.target == other.target
            and self.template.template == other.template.template
            and self.doc_pattern == other.doc_pattern
        )

    def __ne__(self, other):
        return not self == other

    @methodtools.lru_cache(maxsize=None)
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
@dataclass
class SKNMSIRunConfig:
    """Esta clase contiene toda la configuracion necesaria para utilizar alias
    en el metodo run, ademas de crear el objeto result.

    """

    _input: tuple
    _output: tuple
    _result_cls: result.NDResult
    _model_name: str
    _model_type: str

    # initialization

    def __post_init__(self):
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
        """Crea una nueva instancia de configuracion basandose en una
        subclase de SKNMSIMethodABC.

        La clase debe implementar la variable de ''_run_input''
        la cual contiene la configuracion de los alias templates.

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
        _result = method_class._run_result
        _model_name = str(
            getattr(method_class, "_model_name", method_class.__name__)
        )
        _model_type = method_class._model_type

        return cls(
            _model_name=_model_name,
            _model_type=_model_type,
            _input=_input,
            _output=_output,
            _result_cls=_result,
        )

    # API

    @property
    def input_targets(self):
        """Set con todos los parametros configurables de run."""
        return frozenset(patpl.target for patpl in self._input)

    @property
    def output_targets(self):
        """Set con todos los parametros configurables de run output."""
        return frozenset(patpl.target for patpl in self._output)

    @property
    def template_variables(self):
        """Todas las variables de alias-template definidas en todos los \
        targets.

        """
        template_variables = set()
        for patpl in it.chain(self._input, self._output):
            template_variables.update(patpl.template_variables)
        return frozenset(template_variables)

    def make_run_input_alias_map(self, context):
        """Crea un bidict que mapea los alias con los targets.

        El contexto es la configurar que provee el usuario a traves de los
        parametros de init.

        """
        at_map = frozenbidict(
            (patpl.target, patpl.render(context)) for patpl in self._input
        )
        return at_map

    def make_run_output_alias_map(self, context):

        at_map = frozenbidict(
            (patpl.target, patpl.render(context)) for patpl in self._output
        )
        return at_map

    # VALIDATION ==============================================================

    def _parameters_difference(self, method, expected_parameters):
        """Retorna la diferencia entre los parametros de un metodo y un
        conjunto de valores esperados.

        Parameters
        ----------
        method: callable
            Metodo donde se extraeran los parametros.
        expected_parameters: iterable
            Colleccion de parametros que se espera que tenga el metodo

        Returns
        -------
        set :
            Todos los parametros que eran esperados pero que no se encontraban
            en el método.

        """
        parameters = inspect.signature(method).parameters
        parameters_difference = set(expected_parameters).difference(parameters)
        return parameters_difference

    def validate_init_and_run(self, method_class):
        """Valida que los metodos __init__ y run tengan los parametros \
        adecuados.

        El metodo recibe como argumento una clase heredada de SKNMSIMethodABC.

        Raises
        ------
        TypeError :
            Si __init__ o run() no poseen los parametros requeridos.

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
        # reemplazamos cada parametro "target" por uno con un alias
        # los que no son target lo dejamos como vienen
        # el resto de la metadata de loa parametros no cambian
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
        doc = run_method.__doc__ or ""
        for pat in self._input:
            pattern = pat.doc_pattern
            alias = target_alias_map.get(pat.target)
            doc = pattern.sub(alias, doc)
        return doc

    def wrap_run(self, model_instance, run_template_context):
        """Retorna una wrapper para el metodo run.

        Este metodo es utilizado *por cada instancia* de un objeto de clase
        ``SKNMSIMethodABC``.

        El metodo resultante acepta como parámetros los alias resultantes de
        la configuracion y los mapea a los "targets".

        """
        # extraemos el run, calculate_causes y calculate_perceived_positions
        run_method = model_instance.run
        calculate_causes_method = model_instance.calculate_causes

        # creamos el mapeo de alias y target en un diccionario bidireccional
        # para el input y el output
        input_alias_map = self.make_run_input_alias_map(run_template_context)
        output_alias_map = self.make_run_output_alias_map(run_template_context)

        signature_with_alias = self._make_run_signature_with_alias(
            run_method, input_alias_map
        )

        doc_with_alias = self._make_run_doc_with_alias(
            run_method, input_alias_map
        )

        time_range = np.array(model_instance.time_range, dtype=float)
        position_range = np.array(model_instance.position_range, dtype=float)

        time_res = float(model_instance.time_res)
        position_res = float(model_instance.position_res)

        @functools.wraps(run_method)
        def wrapper(*args, **kwargs):

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

            # ya dentro del wrapper lo que hacemos es bindear los parametros
            # asi asignamos los valores de manera correcta a los nombres
            # de parametros adecuados
            bound_params = signature_with_alias.bind(*args, **kwargs)
            bound_params.apply_defaults()

            # ahora separamos los args de los kwargs y cambiamos los
            # alias a los targets correspondientes.
            target_args = bound_params.args
            target_kwargs = {
                input_alias_map.inv.get(k, k): v
                for k, v in bound_params.kwargs.items()
            }
            response, extra = run_method(*target_args, **target_kwargs)
            causes = calculate_causes_method(**response, **extra)

            # now we rename the output
            response_aliased = {
                output_alias_map.get(k, k): v for k, v in response.items()
            }
            extra_aliased = {
                output_alias_map.get(k, k): v for k, v in extra.items()
            }

            return self._result_cls(
                mname=self._model_name,
                mtype=self._model_type,
                nmap=output_alias_map,
                nddata=response_aliased,
                time_range=time_range,
                position_range=position_range,
                time_res=time_res,
                position_res=position_res,
                causes=causes,
                run_params=dict(bound_params.arguments),
                extra=extra_aliased,
            )

        wrapper.__skneuromsi_run_template_context__ = run_template_context
        wrapper.__signature__ = signature_with_alias
        wrapper.__doc__ = doc_with_alias

        return wrapper

    # WRAP INIT================================================================

    def wrap_init(self, init_method):
        signature = inspect.signature(init_method)

        @functools.wraps(init_method)
        def wrapper(instance, *args, **kwargs):
            # primero ejecuto el init_viejo y dejo el objeto configurado
            init_method(instance, *args, **kwargs)

            # bindeamos todos los argumentos al init, asi sabemos bien que
            # valor va con que parametro
            bound_args = signature.bind_partial(instance, *args, **kwargs)
            bound_args.apply_defaults()

            # agarramos los parametros bindeados y si ese parametro, pertenece
            # a una variable de template lo agregamos al diccionario que se
            # va a usar para crear los alias de run()
            run_template_context = {
                tvar: bound_args.arguments[tvar]
                for tvar in self.template_variables
            }

            # creamos el nuevo run, y lo chantamos a nivel de instancia!
            instance.run = self.wrap_run(instance, run_template_context)

        return wrapper

    # MODEL GET-SET STATE =====================================================

    def get_model_state(self, instance):
        state = dict(instance.__dict__)

        # becase the run instance method is a clojure we can serialize this
        # and we only store the context
        state["run"] = dict(instance.run.__skneuromsi_run_template_context__)

        return state

    def set_model_state(self, instance, state):
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
    #
    # instance level redefinition
    ("time_range", "Attribute"),
    ("position_range", "Attribute"),
    ("time_res", "Attribute"),
    ("position_res", "Attribute"),
    ("run", "Method"),
    ("set_random", "Method"),
]

REDEFINE_WITH_DEFAULT = [
    ("_run_result", result.NDResult),
]


class SKNMSIMethodABC:
    """Abstract class that allows to configure method names dynamically."""

    def __init_subclass__(cls):
        """Solo se realizan en este metodo la validacion superfical de
        las subclases, y se delega integramente a SKNMSIRunConfig las
        validaciones propias de la configuracuión.

        """

        # simple validation
        if vars(cls).get("_abstract", False):
            return

        # validate redefinitions
        for attr, attr_type in TO_REDEFINE:
            if not hasattr(cls, attr):
                msg = f"{attr_type} '{attr}' must be redefined"
                raise TypeError(msg)

        # redefine with default
        for attr, default in REDEFINE_WITH_DEFAULT:
            if not hasattr(cls, attr):
                setattr(cls, attr, default)

        # config creation
        config = SKNMSIRunConfig.from_method_class(cls)

        # run and init method vs configuration validation
        config.validate_init_and_run(cls)

        # teardown
        for attr, _ in TO_REDEFINE + REDEFINE_WITH_DEFAULT:
            if attr.startswith("_"):
                delattr(cls, attr)

        cls.__init__ = config.wrap_init(cls.__init__)
        cls._run_config = config

    # def __getstate__(self):
    #     cls = type(self)
    #     return cls._run_config.get_model_state(self)

    # def __setstate__(self, state):
    #     cls = type(self)
    #     cls._run_config.set_model_state(self, state)

    def calculate_causes(self, **kwargs):
        return None
