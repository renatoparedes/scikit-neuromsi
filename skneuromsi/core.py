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
# CLASES Y OTROS CONTENEDORES UTILES
# ===============================================================================


@attr.s(frozen=True, slots=True)
class Stimulus:
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

    metadata = kwargs.pop("metadata", {})
    metadata[_SKCNMSI_METADATA] = _HYPER_PARAMETER

    return attr.ib(init=True, metadata=metadata, **kwargs)


def internal(**kwargs):

    metadata = kwargs.pop("metadata", {})
    metadata[_SKCNMSI_METADATA] = _INTERNAL_VALUE

    kwargs.setdefault("repr", False)

    return attr.ib(init=False, metadata=metadata, **kwargs)


# ===============================================================================
# MODEL DEFINITION
# ===============================================================================


def get_class_fields(cls):

    # vamos a copiar la clase para no destruir la anterior
    # black magic
    cls = type(
        cls.__name__ + "__copied",
        tuple(cls.mro()[1:]),
        vars(cls).copy(),
    )

    # creamos una clase tipo attrs para choriarle los fields
    acls = attr.s(maybe_cls=cls)

    hparams, internals = set(), set()
    for field in attr.fields(acls):
        param_type = field.metadata.get(_SKCNMSI_METADATA)

        if param_type == _HYPER_PARAMETER:
            hparams.add(field.name)

        elif param_type == _INTERNAL_VALUE:
            # at this point we can check if some internal don't have a defaul
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
    signature = inspect.signature(func)

    # for paran_name, param in signature.parameters.items():
    # primero determinamos el tipo, si es un hiper parametro del modelo
    # un valor interno o simplemente un valor del tipo "run"
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

    # separamos los parametros internos y los hiper_parametros
    hparams, internals = get_class_fields(cls)

    # stimuli tiene que ser una lista de callables
    # y vamos a carrgarlo dentro de un diccionario para futuras referencias
    stimuli, run_inputs = {}, set()
    for stimulus in cls.stimuli:

        name = stimulus.__name__
        if not callable(stimulus):
            raise TypeError(f"stimulus '{name}' is not callable")

        # sacamos los parametros del estimulo en hparams e internals
        # y lo que no sabemos que es, lo asumimos como parametro de entrada
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

        # guardamos los run_inputs de todos los stimulos para validarlos
        # contra el integrador despues
        run_inputs.update(sinputs)

    # integration tiene que ser un callable
    if not callable(cls.integration):
        raise TypeError("integration must be callable")
    name = cls.integration.__name__
    ihparams, iinternals, iinputs = get_parameters(
        name, cls.integration, hparams, internals
    )

    # hay que tener en cuenta que el integrador puede tener inputs de run
    # y puede TAMBIEN recibir las salidas de los estimulos. Pero no puede
    # tener run inputs propios.
    diff = iinputs.difference(run_inputs.union(stimuli))
    if diff:
        raise TypeError(f"integration has unknow parameters {', '.join(diff)}")

    integration = Integration(
        name=name,
        hyper_parameters=ihparams,
        internal_values=iinternals,
        stimuli_results=iinputs,
        function=cls.integration,
    )

    # sacamos stimuli e integration del modelo
    del cls.stimuli, cls.integration

    # cargamos los nuevos stimulii en un objeto conf
    conf = Config(stimuli=stimuli, integration=integration)

    # inyectamos la configuracion en la clase
    cls._sknmsi_conf = Config(stimuli=stimuli, integration=integration)

    # el run del modelo lo unico que tiene que hacer es delegat todos los
    # parametros al run de la configuración ademas de pasarse a si mismo
    def run(self, **kwargs):
        return self._sknmsi_conf.run(model=self, inputs=kwargs)

    # ahora hay que enmascarar la firma de run e inyectar run a la clase
    cls.run = change_run_signature(run, conf.run_inputs)

    # convertimos la clase en un clase attrs para que los hiperparametros
    # y valores internos sepan funcionar
    acls = attr.s(maybe_cls=cls)

    # terminamos
    return acls
