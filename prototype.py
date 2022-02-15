import functools
import inspect
import string
from collections.abc import Iterable, Mapping, Collection
from dataclasses import Field, dataclass

from bidict import frozenbidict


# =============================================================================
#
# =============================================================================


@dataclass
class VariableParam:
    target: str
    template: string.Template

    def __post_init__(self):
        self.template = string.Template(self.template)

    @property
    def template_variables(self):
        tpl = self.template
        return {
            match["braced"] for match in tpl.pattern.finditer(tpl.template)
        }


@dataclass
class SKNMSIRunConfig:
    variable_params: tuple

    # initialization

    def __post_init__(self):
        self.variable_params = tuple(self.variable_params)

    @classmethod
    def from_method(cls, method):

        var_params = {}
        for vparam in method._sknms_run_method_config:
            if isinstance(vparam, Mapping):
                vparam = VariableParam(**vparam)
            elif isinstance(vparam, Iterable):
                vparam = VariableParam(*vparam)
            if not isinstance(vparam, VariableParam):
                raise TypeError(
                    f"All elements of {VRP} must be instances of "
                    "'VariableParam' or parameters for their constructor. "
                    f"Found: {vparam!r}"
                )
            if vparam.target in var_params:
                raise ValueError(
                    f"Duplicated VariableParam target {vparam.target}"
                )

            var_params[vparam.target] = vparam

        return cls(variable_params=var_params.values())

    # API

    @property
    def targets(self):
        return frozenset(
            variable_param.target for variable_param in self.variable_params
        )

    @property
    def template_variables(self):
        template_variables = set()
        for variable_param in self.variable_params:
            template_variables.update(variable_param.template_variables)
        return frozenset(template_variables)

    def create_alias_to_target(self, context):
        a2t = {}
        for vparam in self.variable_params:
            a2t[vparam.template.substitute(**context)] = vparam.target
        return frozenbidict(a2t)

    # VALIDATION ==============================================================

    def _get_params_difference(self, expected_parameters, method):
        signature = inspect.signature(method)
        parameters = tuple(signature.parameters)
        targets_not_found = expected_parameters.difference(parameters)
        return targets_not_found

    def validate_init_and_run(self, method):
        targets_not_found = self._get_params_difference(
            self.targets, method.run
        )
        if targets_not_found:
            tnf = ", ".join(targets_not_found)
            raise TypeError(
                f"Target/s {tnf!r} not found as parameter in 'run' method"
            )
        tpl_not_found = self._get_params_difference(
            self.template_variables, method.__init__
        )
        if tpl_not_found:
            tnf = ", ".join(tpl_not_found)
            raise TypeError(
                f"Template variable/s {tnf!r} not found as "
                "parameter in '__init__' method"
            )

    # INIT AND RUN METHODS REPLACEMENT ========================================
    def wrap_run(self, run_method, init_bound_arguments):
        template_bind = {
            k: v
            for k, v in init_bound_arguments.arguments.items()
            if k in self.template_variables
        }
        alias_to_target = self.create_alias_to_target(template_bind)

        run_signature = inspect.signature(run_method)
        aliased_run_parameters = []
        for rparam in run_signature.parameters.values():

            aparam = inspect.Parameter(
                name=alias_to_target.inv[rparam.name],
                kind=rparam.kind,
                default=rparam.default,
                annotation=rparam.annotation,
            )
            aliased_run_parameters.append(aparam)

        new_signature = run_signature.replace(
            parameters=aliased_run_parameters
        )

        @functools.wraps(run_method)
        def new_run(*args, **kwargs):
            bound_params = new_signature.bind(*args, **kwargs)

            target_args = bound_params.args
            target_kwargs = {
                alias_to_target.get(k, k): v
                for k, v in bound_params.kwargs.items()
            }

            return run_method(*target_args, **target_kwargs)

        new_run.__signature__ = new_signature

        return new_run

    def wrap_init(self, init_method):
        signature = inspect.signature(init_method)

        @functools.wraps(init_method)
        def new_init(method_self, *args, **kwargs):
            # primero ejecuto el init_viejo
            init_method(method_self, *args, **kwargs)

            # necesito acceder a los cambios de nombre de parametro que
            # de run que solicita el usuario para este objeto
            # la forma mas facil:
            # 1. bindear (con la funcionalidad de signature) args y kwargs a los parametros de init
            # 2. aplicar los default
            # 3. crear el contexto para los templates de los parametros
            # 4. con eso crear el mapa que relaciona estos parametros nuevos
            # con los verdaderos parametros de run "target".
            # 3 y 4 van a un metodo aparte

            bound_args = signature.bind_partial(method_self, *args, **kwargs)
            bound_args.apply_defaults()
            method_self.run = self.wrap_run(method_self.run, bound_args)

        return new_init


# =============================================================================
# BASES
# =============================================================================


class SKNMSIBase:

    _sknms_abstract = True
    _sknms_run_method_config = None

    def run(self):
        raise NotImplementedError("Default run method has has no implentation")

    def __init_subclass__(cls):

        # simple validation
        if vars(cls).get("_sknms_abstract", False):
            return

        if cls.run is SKNMSIBase.run:
            raise TypeError("'run' method must be redefined")
        if cls._sknms_run_method_config is None:
            raise TypeError(
                f"Class attribute '_sknms_run_method_config' must be redefined"
            )

        # config creation
        config = SKNMSIRunConfig.from_method(cls)

        # run and init method vs configuration validation
        config.validate_init_and_run(cls)

        # teardown
        cls.__init__ = config.wrap_init(cls.__init__)
        cls._sknms_run_method_config = config
