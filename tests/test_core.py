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
