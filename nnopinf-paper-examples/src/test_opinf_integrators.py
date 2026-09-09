import numpy as np
import pytest

from opinf_integrators import (
    crank_nicolson_predict,
    first_order_step,
    get_model_integrator,
    implicit_midpoint_predict,
    predict_continuous_model,
    rk4_predict,
)


class _LinearDecayModel:
    def rhs(self, time, state):
        del time
        return -np.asarray(state)

    def jacobian(self, time, state):
        del time
        return -np.eye(np.asarray(state).size)


def test_rk4_predict_integrates_linear_decay():
    times = np.linspace(0.0, 1.0, 101)
    states = rk4_predict(_LinearDecayModel(), [1.0], times)

    assert np.allclose(states[0], np.exp(-times), atol=1e-9)


@pytest.mark.parametrize(
    ("integration", "expected_predictor"),
    [
        ("rk4", rk4_predict),
        ("crank-nicolson", crank_nicolson_predict),
        ("cn", crank_nicolson_predict),
        ("implicit-midpoint", implicit_midpoint_predict),
    ],
)
def test_predict_continuous_model_dispatches(integration, expected_predictor):
    model = _LinearDecayModel()
    times = np.linspace(0.0, 0.1, 3)

    actual = predict_continuous_model(
        model, [1.0], times, integration=integration
    )
    expected = expected_predictor(model, [1.0], times)

    assert np.allclose(actual, expected)


def test_predict_continuous_model_rejects_unknown_integration():
    with pytest.raises(ValueError, match="integration"):
        predict_continuous_model(
            _LinearDecayModel(), [1.0], [0.0, 0.1], integration="unknown"
        )


def test_get_model_integrator_exact_mapping_and_alias():
    settings = {
        "fom": {"integration": "rk4"},
        "model-integrator": {"OpInf-AH": "cn"},
    }
    assert get_model_integrator(settings, "OpInf-AH") == "crank-nicolson"
    assert get_model_integrator(settings, "OpInf-A") == "rk4"


def test_get_model_integrator_defaults_and_specialized_models():
    assert get_model_integrator({}, "OpInf-AH") == "rk4"
    assert get_model_integrator({}, "LOpInf") == "newmark"
    assert get_model_integrator({}, "entropy") == "rk4"


@pytest.mark.parametrize(
    "settings",
    [
        {"model-integrator": []},
        {"model-integrator": {"OpInf-AH": 3}},
        {"model-integrator": {"OpInf-AH": "unknown"}},
        {"model-integrator": {"LOpInf": "rk4"}},
        {"model-integrator": {"entropy": "implicit-midpoint"}},
    ],
)
def test_get_model_integrator_rejects_invalid_configuration(settings):
    model_type = next(iter(settings.get("model-integrator", {"OpInf-AH": None})), "OpInf-AH")
    with pytest.raises(ValueError):
        get_model_integrator(settings, model_type)


@pytest.mark.parametrize("integration", ["rk4", "crank-nicolson", "implicit-midpoint"])
def test_first_order_step_supports_rhs_without_jacobian(integration):
    state = first_order_step(
        lambda value: -value, [1.0], 0.01, integration=integration
    )
    assert np.allclose(state, np.exp(-0.01), atol=1e-4)
