import attr
import pytest

from skneuromsi import core

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


def test_get_parameters():

    hparams = {"h"}
    internals = {"p"}

    def stim(theta_a, theta_b, h, p):  # implement with mock?
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


def test_stimulus_is_callable():

    try:

        @core.neural_msi_model
        class Foo:
            stimulus_a = 1  # mock?
            stimulus_b = 2
            stimuli = [stimulus_a, stimulus_b]

    except (TypeError, AttributeError):  # maybe fix to overpass AttibuteError
        pass

    else:
        raise TypeError("allows not callable stimuli")


def test_integration_is_callable():

    try:

        @core.neural_msi_model
        class Foo:
            integration = 3  # mock?

    except (TypeError, AttributeError):  # maybe fix to overpass AttibuteError
        pass

    else:
        raise TypeError("allows not callable integration")


# def test_integration_unknown_pars():  # TODO add stimuli
#    def integration(theta_a, theta_b, h, p):  # implement with mock?
#        theta_a = 5
#        theta_b = 7
#        return theta_a * h + theta_b * p

#    try:

#        @core.neural_msi_model
#        class Foo:
#            integration = integration

#    except (TypeError):
#        pass

#    else:
#        raise TypeError("integration allows unknown parameters")


# def test_remove_stimuli():

# def test_remove_integration():

# =============================================================================
# CLASES Y OTROS CONTENEDORES UTILES
# =============================================================================


def test_stimulus_attrib():
    def stim(theta_a, theta_b, h, p):  # implement with mock?
        theta_a = 5
        theta_b = 7
        return theta_a * h + theta_b * p

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
    def stim(theta_a, theta_b, h, p):  # implement with mock?
        theta_a = 5
        theta_b = 7
        return theta_a * h + theta_b * p

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


def test_stimulus_frozen():  # Maybe a shorter way to do this?
    def stim(theta_a, theta_b, h, p):
        theta_a = 5
        theta_b = 7
        return theta_a * h + theta_b * p

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

    try:
        new_stim.name = "stim"
    except attr.exceptions.FrozenInstanceError:
        pass
    else:
        raise TypeError("class Stimulus is not frozen")


def test_integration_attrib():
    def integration(theta_a, theta_b, h, p):  # implement with mock?
        theta_a = 5
        theta_b = 7
        return theta_a * h + theta_b * p

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
    def integration(theta_a, theta_b, h, p):  # implement with mock?
        theta_a = 5
        theta_b = 7
        return theta_a * h + theta_b * p

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
    def integration(theta_a, theta_b, h, p):  # implement with mock?
        theta_a = 5
        theta_b = 7
        return theta_a * h + theta_b * p

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

    try:
        new_integration.name = "integration"
    except attr.exceptions.FrozenInstanceError:
        pass
    else:
        raise TypeError("class Integration is not frozen")


# test_config():
