import pytest

from skneuromsi import core


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


# =============================================================================
# CLASES Y OTROS CONTENEDORES UTILES
# =============================================================================


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

    assert new_stim.hyper_parameters == {"h"}
    assert new_stim.internal_values == {"p"}
    assert new_stim.run_inputs == {"theta_a", "theta_b"}


# def test_stimulus_is_callable():
#    assert not callable(...)

# def test_stimulus_name():
#    ...

# def test_integration_property():

# def test_integration_is_callable():
