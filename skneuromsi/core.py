#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-neuromsi Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""
Implementation of multisensory integration neurocomputational models in Python.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import inspect
import itertools as it

import attr
from attr import validators as vldt

# ===============================================================================
# CONSTANTS
# ===============================================================================

_SKCNMSI_METADATA = "__skc_neuro_msi__"


_HYPER_PARAMETER = type("_HYPER_PARAMETER", (object,), {})

_INTERNAL_VALUE = type("_INTERNAL_VALUE", (object,), {})


# ===============================================================================
# CLASSES AND USEFUL CONTAINERS
# ===============================================================================


@attr.s(frozen=True, slots=True)
class Stimulus:
    """Class for computing unisensory estimators.

    Attributes
    ----------
    name: ``attr.ib``
        Name of the estimator.
    hyper_parameters: ``attr.ib``
        Set of hyperparameters coming from a neural_msi_model.
    internal_values: ``attr.ib``
        Set of internal values coming from a neural_msi_model.
    run_inputs: ``attr.ib``
        Set of inputs coming from a neural_msi_model run.
    function: ``attr.ib``
        Callable that defines the computation of the unisensory estimate.
    """

    name = attr.ib(converter=str)
    hyper_parameters = attr.ib(converter=set)
    internal_values = attr.ib(converter=set)
    run_inputs = attr.ib(converter=set)
    function = attr.ib(validator=vldt.is_callable())

    @property
    def parameters(self):
        return set(
            it.chain(
                self.hyper_parameters,
                self.internal_values,
                self.run_inputs,
            )
        )


@attr.s(frozen=True, slots=True)
class Integration:
    """Class for computing the multisensory estimator.

    Attributes
    ----------
    name: ``attr.ib``
        Name of the estimator.
    hyper_parameters: ``attr.ib``
        Set of hyperparameters coming from a neural_msi_model.
    internal_values: ``attr.ib``
        Set of internal values coming from a neural_msi_model.
    stimuli_results: ``attr.ib``
        Set of inputs coming from unisensory estimators.
    function: ``attr.ib``
        Callable that defines the computation of the multisensory estimate.
    """

    name = attr.ib(converter=str)
    hyper_parameters = attr.ib(converter=set)
    internal_values = attr.ib(converter=set)
    stimuli_results = attr.ib(converter=set)
    function = attr.ib(validator=vldt.is_callable())

    @property
    def parameters(self):
        return set(
            it.chain(
                self.hyper_parameters,
                self.internal_values,
                self.stimuli_results,
            )
        )


@attr.s(frozen=True, slots=True)
class Config:
    """Class for configuring a neural_msi_model.

    Attributes
    ----------
    stimuli: ``attr.ib``
        List of skneuromsi.Stimulus that define the unisensory
        estimators of the neural_msi_model.
    integration: ``attr.ib``
        A skneuromsi.Integration that defines the multisensory
        estimator of the neural_msi_model.
    run_inputs: ``attr.ib``
        Set of inputs coming from a neural_msi_model run.

    Methods
    -------
    get_model_values(model)
        Gets the hyperparameters and internals of the neural_msi_model.
    run(model, inputs)
        Executes the multisensory integration.

    """

    stimuli = attr.ib()
    integration = attr.ib()
    run_inputs = attr.ib(init=False)

    @run_inputs.default
    def _allinputs_default(self):
        inputs = set()
        for s in self.stimuli.values():
            inputs.update(s.run_inputs)
        return inputs

    def _validate_inputs(self, model, inputs):
        expected = set(self.run_inputs)  # sacamos una copia

        # por las dudas se usa en dos lugares
        run_name = f"{type(model).__name__}.run()"

        for rinput in inputs:
            if rinput not in expected:
                raise TypeError(
                    f"{run_name} got an unexpected keyword argument '{rinput}'"
                )
            expected.remove(rinput)

        if expected:  # si algo sobro
            required = ", ".join(f"'{e}'" for e in expected)
            raise TypeError(
                f"{run_name} missing required argument/s: {required}"
            )

    def get_model_values(self, model):
        def flt(attribute, _):
            return attribute.metadata.get(_SKCNMSI_METADATA) in (
                _HYPER_PARAMETER,
                _INTERNAL_VALUE,
            )

        return attr.asdict(model, filter=flt)

    def run(self, model, inputs):
        # validamos que no falte ni sobre ningun input
        self._validate_inputs(model, inputs)

        # extraemos todos los hiperparametros y valores internos
        model_values = self.get_model_values(model)

        # mezclamos todos los inputs con todos los demas parametros
        # en un solo espacio de nombre
        namespace = dict(**inputs, **model_values)

        # resolvemos todos los estimulos
        stimuli_results = {}
        for stimulus in self.stimuli.values():
            kwargs = {k: namespace[k] for k in stimulus.parameters}
            stimuli_results[stimulus.name] = stimulus.function(**kwargs)

        # ahora agregamos al namespace los valores resultantes de los estimulos
        namespace.update(stimuli_results)

        # y ejecutamos el integrador
        kwargs = {k: namespace[k] for k in self.integration.parameters}
        result = self.integration.function(**kwargs)

        return result


# =============================================================================
# FUNCTIONS
# =============================================================================


def hparameter(**kwargs):
    """Creates an hyperparameter attribute.

    Parameters
    ----------
    **kwargs: ``dict``, optional
        Extra arguments for the hyperparameter setup.

    Returns
    ----------
    ``attr.ib``
        Hyperparameter attribute.
    """

    metadata = kwargs.pop("metadata", {})
    metadata[_SKCNMSI_METADATA] = _HYPER_PARAMETER

    return attr.ib(init=True, metadata=metadata, **kwargs)


def internal(**kwargs):
    """Creates an internal attribute.

    Parameters
    ----------
    **kwargs: ``dict``, optional
        Extra arguments for the internal setup.

    Returns
    ----------
    ``attr.ib``
        Internal attribute.
    """

    metadata = kwargs.pop("metadata", {})
    metadata[_SKCNMSI_METADATA] = _INTERNAL_VALUE

    kwargs.setdefault("repr", False)

    return attr.ib(init=False, metadata=metadata, **kwargs)


# ===============================================================================
# MODEL DEFINITION
# ===============================================================================


def get_class_fields(cls):
    """Gets the fields of a class.

    Parameters
    ----------
    cls: ``class``
        Class object relevant for the model, usually built on top of
        an estimator.

    Returns
    ----------
    hparams: ``set``
        Set containing the class attributes labeled as hyperparameters.
    internals: ``set``
        Set containing the class attributes labeled as internals.
    """

    # copy the class to avoid destroying the original
    # black magic
    cls = type(
        cls.__name__ + "__copied",
        tuple(cls.mro()[1:]),
        vars(cls).copy(),
    )

    # create an attrs class to get fields from
    acls = attr.s(maybe_cls=cls)

    hparams, internals = set(), set()
    for field in attr.fields(acls):
        param_type = field.metadata.get(_SKCNMSI_METADATA)

        if param_type == _HYPER_PARAMETER:
            hparams.add(field.name)

        elif param_type == _INTERNAL_VALUE:
            # we check if any internal doesn't have a default
            # this is a problem
            if field.default is attr.NOTHING:
                raise TypeError(f"internal '{field.name}' must has a default")

            internals.add(field.name)

        else:
            raise TypeError(
                f"field '{field.name}' is neither hparameter() or internal()"
            )

    return hparams, internals


def get_parameters(name, func, hyper_parameters, internal_values):
    """Classifies the parameters of a function in hyperameters,
    internals or run inputs.

    Parameters
    ----------
    name: ``str``
        Name of the function.
    func: ``callable``
        Function to extract parameters from, usually an estimator.
    hyper_parameters: ``set``
        Set containing attributes labeled as hyperparameters.
    internal_values: ``set``
        Set containing the attributes labeled as internals.

    Returns
    ----------
    shparams: ``set``
        Set containing the function parameters classified as hyperparameters.
    sinternals: ``set``
        Set containing the function parameters classified as internals.
    sinputs: ``set``
        Set containing the function parameters classified as run inputs.
    """

    signature = inspect.signature(func)

    # for paran_name, param in signature.parameters.items():
    # first determine whether it is an hyperparameter, an internal
    # or a run input.
    shparams, sinternals, sinputs = set(), set(), set()
    for param_name, param_value in signature.parameters.items():
        if param_value.default is not param_value.empty:
            raise TypeError(
                "No default value alowed in stimuli and integration. "
                f"Function '{name} -> {param_name}={param_value.default}'"
            )
        if param_name in hyper_parameters:
            shparams.add(param_name)
        elif param_name in internal_values:
            sinternals.add(param_name)
        else:
            sinputs.add(param_name)

    return shparams, sinternals, sinputs


def change_run_signature(run, run_inputs):
    """Modifies the signature of the run method of a neural_msi_model.

    Parameters
    ----------
    run: ``callable``
        Function that delegates all the parameters to the run method of
        a skneuromsi.Config class.
    run_inputs: ``set``
        Set containing the class attributes labeled as run inputs.

    Returns
    ----------
    run: ``callable``
        Run method with a new signature including the run_input parameters.
    """

    signature = inspect.signature(run)

    self_param = signature.parameters["self"]
    new_params = [self_param] + [
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
        for name in run_inputs
    ]

    new_signature = signature.replace(parameters=new_params)

    run.__signature__ = new_signature

    return run


def neural_msi_model(cls):
    """Defines a class as a neural_msi_model.

    Parameters
    ----------
    cls: ``class``
        Class object of the model.

    Returns
    ----------
    acls: ``attr.s``
        Class with a neural_msi_model setup.
    """

    # classify the class attributes as hyperparameters or internals
    hparams, internals = get_class_fields(cls)

    # put the estimators functions inside a dictionary
    stimuli, run_inputs = {}, set()
    for stimulus in cls.stimuli:

        name = stimulus.__name__
        if not callable(stimulus):
            raise TypeError(f"stimulus '{name}' is not callable")

        # classify the parameters of the estimators
        shparams, sinternals, sinputs = get_parameters(
            name, stimulus, hparams, internals
        )

        stimuli[name] = Stimulus(
            name=name,
            hyper_parameters=shparams,
            internal_values=sinternals,
            run_inputs=sinputs,
            function=stimulus,
        )

        # save the run_inputs of the unisensory estimators to be validated
        # against the multisensory estimator afterwards.
        run_inputs.update(sinputs)

    # the multisensory estimator must be a callable
    if not callable(cls.integration):
        raise TypeError("integration must be callable")
    name = cls.integration.__name__
    ihparams, iinternals, iinputs = get_parameters(
        name, cls.integration, hparams, internals
    )

    # the multisensory esitmator may have run inputs and receive the
    # unisensory estimators results but cannot have run inputs of its own
    diff = iinputs.difference(run_inputs.union(stimuli))
    if diff:
        raise TypeError(
            f"integration has unknown parameters {', '.join(diff)}"
        )

    integration = Integration(
        name=name,
        hyper_parameters=ihparams,
        internal_values=iinternals,
        stimuli_results=iinputs,
        function=cls.integration,
    )

    # remove the stimuli e integration attributes from the model
    del cls.stimuli, cls.integration

    # load the new stimuli in an config object
    conf = Config(stimuli=stimuli, integration=integration)

    # include the config object inside the class
    cls._sknmsi_conf = Config(stimuli=stimuli, integration=integration)

    # define a run method that delegates all the parameters to the
    # run method of the config object.
    def run(self, **kwargs):
        return self._sknmsi_conf.run(model=self, inputs=kwargs)

    # masks the signature of the run method and include it inside
    # the class
    cls.run = change_run_signature(run, conf.run_inputs)

    # convert the class to an attrs class so that the hyperparameters
    # and internals work.
    acls = attr.s(maybe_cls=cls)

    return acls
