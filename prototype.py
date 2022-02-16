import functools
import inspect
import string
from collections.abc import Iterable, Mapping, Collection
from dataclasses import Field, dataclass
from attr import frozen

from bidict import frozenbidict


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

    Attributes
    ----------
    template_variables: frozenset
        Lista todas las variables que aparecen en el template.

    """

    target: str
    template: string.Template

    def __post_init__(self) -> None:
        if not isinstance(self.template, string.Template):
            self.template = string.Template(self.template)

    @property
    def template_variables(self) -> frozenset:
        """Variables del template."""
        tpl = self.template
        return frozenset(
            match["braced"] for match in tpl.pattern.finditer(tpl.template)
        )

    def render(self, context) -> str:
        """Crea un nuevo alias basado en el contexto provisto."""
        return self.template.substitute(context)


@dataclass
class SKNMSIRunConfig:
    parameter_alias_templates: tuple

    # initialization

    def __post_init__(self):
        self.parameter_alias_templates = tuple(self.parameter_alias_templates)

    @classmethod
    def from_method_class(cls, method):
        """Crea una nueva instancia de configuracion basandose en una
        subclase de SKNMSIMethodABC.

        La clase debe implementar la variable de ''_sknms_run_method_config''
        la cual contiene la configuracion de los alias templates.

        """

        parameter_alias_templates = {}
        for patpl in method._sknms_run_method_config:

            if isinstance(patpl, Mapping):
                patpl = ParameterAliasTemplate(**patpl)

            elif isinstance(patpl, Iterable):
                patpl = ParameterAliasTemplate(*patpl)

            if not isinstance(patpl, ParameterAliasTemplate):
                raise TypeError(
                    "All elements of '_sknms_run_method_config' must be "
                    "instances of  'ParameterAliasTemplate' or parameters for "
                    f"their constructor. Found: {patpl!r}"
                )

            if patpl.target in parameter_alias_templates:
                raise ValueError(
                    f"Duplicated ParameterAliasTemplate target {patpl.target}"
                )

            parameter_alias_templates[patpl.target] = patpl

        return cls(parameter_alias_templates.values())

    # API

    @property
    def targets(self):
        """Set con todos los parametros configurables de run."""
        return frozenset(
            patpl.target for patpl in self.parameter_alias_templates
        )

    @property
    def template_variables(self):
        """Todas las variables de alias-template definidas en todos los \
        targets.

        """
        template_variables = set()
        for patpl in self.parameter_alias_templates:
            template_variables.update(patpl.template_variables)
        return frozenset(template_variables)

    def create_alias_target_map(self, context):
        """Crea un diccionario bidireccional que mapea los alias con los \
        targets.

        El contexto es la configurar que provee el usuario a traves de los
        parametros de init.

        """
        at_map = frozenbidict(
            (patpl.render(context), patpl.target)
            for patpl in self.parameter_alias_templates
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
            method_class.run, self.targets
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

    # INIT AND RUN METHODS REPLACEMENT ========================================
    def wrap_run(self, run_method, run_parameters_template_context):
        """Retorna una wrapper para el metodo run.

        Este metodo es utilizado *por cada instancia* de un objeto de clase
        ``SKNMSIMethodABC``.

        El metodo resultante acepta como parámetros los alias resultantes de
        la configuracion y los mapea a los "targets".

        """

        # creamos el mapeo de alias y target en un diccionario bidireccional
        alias_target_map = self.create_alias_target_map(
            run_parameters_template_context
        )

        # reemplazamos cada parametro "target" por uno con un alias
        # los que no son target lo dejamos como vienen
        # el resto de la metadata de loa parametros no cambian
        run_signature = inspect.signature(run_method)
        aliased_run_parameters = []
        for run_parameter in run_signature.parameters.values():

            alias_parameter = inspect.Parameter(
                name=alias_target_map.inv.get(
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

        @functools.wraps(run_method)
        def wrapper(*args, **kwargs):

            # ya dentro del wrapper lo que hacemos es bindear los parametros
            # asi asignamos los valores de manera correcta a los nombres
            # de parametros adecuados
            bound_params = signature_with_alias.bind(*args, **kwargs)

            # ahora separamos los args de los kwargs y cambiamos los
            # alias a los targets correspondientes.
            target_args = bound_params.args
            target_kwargs = {
                alias_target_map.get(k, k): v
                for k, v in bound_params.kwargs.items()
            }

            return run_method(*target_args, **target_kwargs)

        wrapper.__signature__ = signature_with_alias

        return wrapper

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
            run_parameters_template_context = {
                k: v
                for k, v in bound_args.arguments.items()
                if k in self.template_variables
            }

            # creamos el nuevo run, y lo chantamos a nivel de instancia!
            instance.run = self.wrap_run(
                instance.run, run_parameters_template_context
            )

        return wrapper


# =============================================================================
# BASES
# =============================================================================


class SKNMSIMethodABC:

    _sknms_abstract = True
    _sknms_run_method_config = None

    def run(self):
        raise NotImplementedError("Default run method has has no implentation")

    def __init_subclass__(cls):

        # simple validation
        if vars(cls).get("_sknms_abstract", False):
            return

        if cls.run is SKNMSIMethodABC.run:
            raise TypeError("'run' method must be redefined")
        if cls._sknms_run_method_config is None:
            raise TypeError(
                f"Class attribute '_sknms_run_method_config' must be redefined"
            )

        # config creation
        config = SKNMSIRunConfig.from_method_class(cls)

        # run and init method vs configuration validation
        config.validate_init_and_run(cls)

        # teardown
        cls.__init__ = config.wrap_init(cls.__init__)
        cls._sknms_run_method_config = config
