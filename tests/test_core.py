import attr

import pytest

from skneuromsi import core


# ===============================================================================
# COMMONN
# ===============================================================================


@attr.s
class Thing:
    name = attr.ib()

    @property
    def __name__(self):
        return self.name


# =============================================================================
# FUNCTIONS
# =============================================================================


def test_hyperparameter():
    hparam = core.hparameter()
    assert hparam.metadata[core._SKCNMSI_METADATA] == core._HYPER_PARAMETER
    assert hparam.init


def test_internal():
    internal = core.internal()
    assert internal.metadata[core._SKCNMSI_METADATA] == core._INTERNAL_VALUE
    assert not internal.repr
    assert not internal.init


def test_internal_init_True():
    with pytest.raises(TypeError):
        core.internal(init=True)


# =============================================================================
# neural_msi_model
# =============================================================================


def test_get_class_field():
    class Foo:
        h = core.hparameter()
        p = core.internal(default=None)

    hparams, internals = core.get_class_fields(Foo)

    assert hparams == {"h"}
    assert internals == {"p"}


def test_get_class_field_no_internal_default():
    class Foo:
        p = core.internal()

    with pytest.raises(TypeError):
        core.get_class_fields(Foo)


def test_get_class_field_call_two_times():
    class Foo:
        h = core.hparameter()
        p = core.internal(default=None)

    core.get_class_fields(Foo)
    hparams, internals = core.get_class_fields(Foo)

    assert hparams == {"h"}
    assert internals == {"p"}


def test_get_class_fields_no_hiper_no_internal():
    class Foo:
        j = attr.ib()  # Works only with attr?

    with pytest.raises(TypeError):
        core.get_class_fields(Foo)


def test_get_parameters():  # TODO create more tests for get_parameters

    hparams = {"h"}
    internals = {"p"}

    def stim(theta_a, theta_b, h, p):  # implement property based test?
        theta_a = 5
        theta_b = 7
        return theta_a * h + theta_b * p

    name = stim.__name__
    shparams, sinternals, sinputs = core.get_parameters(
        name, stim, hparams, internals
    )

    assert shparams == {"h"}
    assert sinternals == {"p"}
    assert sinputs == {"theta_a", "theta_b"}


def test_get_parameters_no_default():

    hparams = {"h"}
    internals = {"p"}

    def stim(h, p, theta_a=5, theta_b=7):
        return theta_a * h + theta_b * p

    name = stim.__name__

    with pytest.raises(TypeError):
        shparams, sinternals, sinputs = core.get_parameters(
            name, stim, hparams, internals
        )


def test_stimulus_is_callable():

    a, b = Thing("a"), Thing("b")

    with pytest.raises(TypeError):

        @core.neural_msi_model
        class Foo:
            stimulus_a = a
            stimulus_b = b
            stimuli = [stimulus_a, stimulus_b]


def test_integration_is_callable():
    def stim():
        return "something"

    c = Thing("c")

    with pytest.raises(TypeError):

        @core.neural_msi_model
        class Foo:
            stimulus_a = stim
            stimulus_b = stim
            stimuli = [stimulus_a, stimulus_b]
            integration = c


def test_integration_unknown_pars():
    def stim():
        return "something"

    def ms_integration(theta_a, theta_b, h, p, j):
        theta_a = 5
        theta_b = 7
        return theta_a * h + theta_b * p + j

    with pytest.raises(TypeError):

        @core.neural_msi_model
        class Foo:
            stimulus_a = stim
            stimulus_b = stim
            stimuli = [stimulus_a, stimulus_b]
            integration = ms_integration


def test_remove_stimuli():
    def stim(theta_a, theta_b):
        theta_a = 5
        theta_b = 7
        return theta_a + theta_b

    def ms_integration(theta_a, theta_b):
        h = 2
        p = 1
        return theta_a * h + theta_b * p

    @core.neural_msi_model
    class Foo:
        stimulus_a = stim
        stimulus_b = stim
        stimuli = [stimulus_a, stimulus_b]
        integration = ms_integration

    with pytest.raises(AttributeError):
        Foo.stimuli


def test_remove_integration():
    def stim(theta_a, theta_b):
        theta_a = 5
        theta_b = 7
        return theta_a + theta_b

    def ms_integration(theta_a, theta_b):
        h = 2
        p = 1
        return theta_a * h + theta_b * p

    @core.neural_msi_model
    class Foo:
        stimulus_a = stim
        stimulus_b = stim
        stimuli = [stimulus_a, stimulus_b]
        integration = ms_integration

    with pytest.raises(AttributeError):
        Foo.integration


# =============================================================================
# CLASES Y OTROS CONTENEDORES UTILES
# =============================================================================


def test_stimulus_attrib():
    def stim():
        return "something"

    shparams = {"h"}
    sinternals = {"p"}
    sinputs = {"theta_a", "theta_b"}
    name = stim.__name__

    new_stim = core.Stimulus(
        name=name,
        hyper_parameters=shparams,
        internal_values=sinternals,
        run_inputs=sinputs,
        function=stim,
    )

    assert new_stim.name == stim.__name__
    assert new_stim.hyper_parameters == {"h"}
    assert new_stim.internal_values == {"p"}
    assert new_stim.run_inputs == {"theta_a", "theta_b"}
    assert new_stim.function == stim


def test_stimulus_property():
    def stim():
        return "something"

    shparams = {"h"}
    sinternals = {"p"}
    sinputs = {"theta_a", "theta_b"}
    name = stim.__name__

    new_stim = core.Stimulus(
        name=name,
        hyper_parameters=shparams,
        internal_values=sinternals,
        run_inputs=sinputs,
        function=stim,
    )

    assert new_stim.parameters == {"h", "p", "theta_a", "theta_b"}


def test_stimulus_frozen():
    def stim():
        return "something"

    shparams = {"h"}
    sinternals = {"p"}
    sinputs = {"theta_a", "theta_b"}
    name = stim.__name__

    new_stim = core.Stimulus(
        name=name,
        hyper_parameters=shparams,
        internal_values=sinternals,
        run_inputs=sinputs,
        function=stim,
    )

    with pytest.raises(attr.exceptions.FrozenInstanceError):
        new_stim.name = "stim"


def test_integration_attrib():
    def integration():
        return "something"

    ihparams = {"h"}
    iinternals = {"p"}
    iinputs = {"theta_a", "theta_b"}
    name = integration.__name__

    new_integration = core.Integration(
        name=name,
        hyper_parameters=ihparams,
        internal_values=iinternals,
        stimuli_results=iinputs,
        function=integration,
    )

    assert new_integration.name == integration.__name__
    assert new_integration.hyper_parameters == {"h"}
    assert new_integration.internal_values == {"p"}
    assert new_integration.stimuli_results == {"theta_a", "theta_b"}
    assert new_integration.function == integration


def test_integration_property():
    def integration():
        return "something"

    ihparams = {"h"}
    iinternals = {"p"}
    iinputs = {"theta_a", "theta_b"}
    name = integration.__name__

    new_integration = core.Integration(
        name=name,
        hyper_parameters=ihparams,
        internal_values=iinternals,
        stimuli_results=iinputs,
        function=integration,
    )

    assert new_integration.parameters == {"h", "p", "theta_a", "theta_b"}


def test_integration_frozen():
    def integration():
        return "something"

    ihparams = {"h"}
    iinternals = {"p"}
    iinputs = {"theta_a", "theta_b"}
    name = integration.__name__

    new_integration = core.Integration(
        name=name,
        hyper_parameters=ihparams,
        internal_values=iinternals,
        stimuli_results=iinputs,
        function=integration,
    )

    with pytest.raises(attr.exceptions.FrozenInstanceError):
        new_integration.name = "integration"


def test_config_attrib():
    def stim():
        return "something"

    shparams = {"h"}
    sinternals = {"p"}
    sinputs = {"theta_a", "theta_b"}
    name = stim.__name__

    new_stim = core.Stimulus(
        name=name,
        hyper_parameters=shparams,
        internal_values=sinternals,
        run_inputs=sinputs,
        function=stim,
    )

    def integration():
        return "something"

    ihparams = {"h"}
    iinternals = {"p"}
    iinputs = {"theta_a", "theta_b"}
    name = integration.__name__

    new_integration = core.Integration(
        name=name,
        hyper_parameters=ihparams,
        internal_values=iinternals,
        stimuli_results=iinputs,
        function=integration,
    )

    stims = {"stim": new_stim}
    conf = core.Config(stimuli=stims, integration=new_integration)

    assert conf.stimuli == stims
    assert conf.integration == new_integration
    assert conf.run_inputs == sinputs


def test_config_frozen():
    def stim():
        return "something"

    shparams = {"h"}
    sinternals = {"p"}
    sinputs = {"theta_a", "theta_b"}
    name = stim.__name__

    new_stim = core.Stimulus(
        name=name,
        hyper_parameters=shparams,
        internal_values=sinternals,
        run_inputs=sinputs,
        function=stim,
    )

    def integration():
        return "something"

    ihparams = {"h"}
    iinternals = {"p"}
    iinputs = {"theta_a", "theta_b"}
    name = integration.__name__

    new_integration = core.Integration(
        name=name,
        hyper_parameters=ihparams,
        internal_values=iinternals,
        stimuli_results=iinputs,
        function=integration,
    )

    stims = {"stim": new_stim}
    conf = core.Config(stimuli=stims, integration=new_integration)

    with pytest.raises(attr.exceptions.FrozenInstanceError):
        conf.run_inputs = {"a", "b", "c"}


def test_config_validate_inputs_unexpected_arg():
    def stim():
        return "something"

    shparams = {"h"}
    sinternals = {"p"}
    sinputs = {"theta_a", "theta_b"}
    name = stim.__name__

    new_stim = core.Stimulus(
        name=name,
        hyper_parameters=shparams,
        internal_values=sinternals,
        run_inputs=sinputs,
        function=stim,
    )

    def integration():
        return "something"

    ihparams = {"h"}
    iinternals = {"p"}
    iinputs = {"theta_a", "theta_b"}
    name = integration.__name__

    new_integration = core.Integration(
        name=name,
        hyper_parameters=ihparams,
        internal_values=iinternals,
        stimuli_results=iinputs,
        function=integration,
    )

    stims = {"stim": new_stim}

    class Thing:
        pass

    a = Thing()
    a.__name__ = "a"

    conf = core.Config(stimuli=stims, integration=new_integration)

    with pytest.raises(TypeError):
        conf._validate_inputs(
            model=a, inputs={"theta_a", "theta_b", "theta_c"}
        )


def test_config_validate_inputs_missing_arg():
    def stim():
        return "something"

    shparams = {"h"}
    sinternals = {"p"}
    sinputs = {"theta_a", "theta_b"}
    name = stim.__name__

    new_stim = core.Stimulus(
        name=name,
        hyper_parameters=shparams,
        internal_values=sinternals,
        run_inputs=sinputs,
        function=stim,
    )

    def integration():
        return "something"

    ihparams = {"h"}
    iinternals = {"p"}
    iinputs = {"theta_a", "theta_b"}
    name = integration.__name__

    new_integration = core.Integration(
        name=name,
        hyper_parameters=ihparams,
        internal_values=iinternals,
        stimuli_results=iinputs,
        function=integration,
    )

    stims = {"stim": new_stim}

    class Thing:
        pass

    a = Thing()
    a.__name__ = "a"

    conf = core.Config(stimuli=stims, integration=new_integration)

    with pytest.raises(TypeError):
        conf._validate_inputs(model=a, inputs={"theta_a"})


def test_config_get_model_values():
    def stim():
        return "something"

    shparams = {"h"}
    sinternals = {"p"}
    sinputs = {"theta_a", "theta_b"}
    name = stim.__name__

    new_stim = core.Stimulus(
        name=name,
        hyper_parameters=shparams,
        internal_values=sinternals,
        run_inputs=sinputs,
        function=stim,
    )

    def integration():
        return "something"

    ihparams = {"h"}
    iinternals = {"p"}
    iinputs = {"theta_a", "theta_b"}
    name = integration.__name__

    new_integration = core.Integration(
        name=name,
        hyper_parameters=ihparams,
        internal_values=iinternals,
        stimuli_results=iinputs,
        function=integration,
    )

    stims = {"stim": new_stim}

    @attr.s
    class Foo:
        hparam = core.hparameter(default=4.0)
        internal = core.internal(default=3.0)

    conf = core.Config(stimuli=stims, integration=new_integration)
    model = Foo()

    assert conf.get_model_values(model) == {"hparam": 4.0, "internal": 3.0}


def test_run():
    def stimulus_a(a, h):
        return a + 1 * h

    def integ(stimulus_a):
        return stimulus_a + 1

    @core.neural_msi_model
    class Example:

        # hiper parameters
        h = core.hparameter()

        # estimulii

        stimuli = [stimulus_a]
        integration = integ

    model = Example(h=5)

    assert model.run(a=25) == (25 + 1 * 5) + 1
