import numpy as np
import pytest

import energy_preserving_opinf as epopinf
from energy_preserving_opinf import (
    _EnergyPreservingFitWorkspace,
    _SkewQuadraticMap,
    energy_preserving_constraint_matrix,
    energy_preserving_residual,
    fit_energy_preserving_model,
    interpolate_energy_preserving_models,
)


def _training_data(constant=None, linear=None):
    states = np.random.default_rng(4).normal(size=(2, 40))
    quadratic = np.vstack((states[0] ** 2, states[0] * states[1], states[1] ** 2))
    # This compressed quadratic operator satisfies x.T H(x kron x) = 0.
    H = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    ddts = H @ quadratic
    if linear is not None:
        ddts += linear @ states
    if constant is not None:
        ddts += constant[:, None]
    return states, ddts, H


@pytest.mark.parametrize(
    ("model_form", "constant", "linear"),
    [
        ("H", None, None),
        ("AH", None, np.array([[0.2, -0.1], [0.3, -0.4]])),
        (
            "cAH",
            np.array([0.7, -0.5]),
            np.array([[0.2, -0.1], [0.3, -0.4]]),
        ),
    ],
)
def test_fit_energy_preserving_model(model_form, constant, linear):
    states, ddts, H = _training_data(constant, linear)
    model = fit_energy_preserving_model(states, ddts, model_form=model_form)

    assert np.allclose(model.H_.entries, H)
    assert energy_preserving_residual(model.H_.entries) < 1e-12
    if linear is not None:
        assert np.allclose(model.A_.entries, linear)
    if constant is not None:
        assert np.allclose(model.c_.entries, constant)


def test_cah_constraint_matrix_accounts_for_constant_and_linear_blocks():
    constraints = energy_preserving_constraint_matrix(2, "cAH")
    assert constraints.shape == (4, 12)
    assert np.all(constraints[:, :3].toarray() == 0)
    assert np.all(constraints[:, 6:9].toarray() == 0)


@pytest.mark.parametrize("state_dimension", [1, 2, 4])
def test_energy_preserving_residual_matches_constraint_matrix(state_dimension):
    generator = np.random.default_rng(state_dimension)
    quadratic_dimension = state_dimension * (state_dimension + 1) // 2
    entries = generator.normal(size=(state_dimension, quadratic_dimension))
    expected = np.max(
        np.abs(
            energy_preserving_constraint_matrix(state_dimension, "H")
            @ entries.ravel()
        ),
        initial=0.0,
    )

    assert np.isclose(energy_preserving_residual(entries), expected)


def test_interpolate_energy_preserving_cah_models_at_training_parameters():
    linear = np.array([[0.2, -0.1], [0.3, -0.4]])
    constants = (np.array([0.7, -0.5]), np.array([-0.2, 0.8]))
    models = []
    for constant in constants:
        states, ddts, _ = _training_data(constant, linear)
        models.append(
            fit_energy_preserving_model(states, ddts, model_form="cAH")
        )

    parameters = np.array([0.0, 1.0])
    interpolated = interpolate_energy_preserving_models(
        parameters, models, model_form="cAH"
    )
    state = np.array([0.25, -0.4])
    for parameter, model in zip(parameters, models):
        assert np.allclose(
            interpolated.rhs(0.0, parameter, state), model.rhs(0.0, state)
        )


@pytest.mark.parametrize("state_dimension", [2, 3, 5])
def test_skew_quadratic_map_is_energy_preserving_and_has_correct_adjoint(
    state_dimension,
):
    generator = np.random.default_rng(state_dimension)
    mapping = _SkewQuadraticMap(state_dimension)
    parameters = generator.normal(size=mapping.parameter_dimension)
    entry_gradient = generator.normal(
        size=(state_dimension, state_dimension * (state_dimension + 1) // 2)
    )

    entries = mapping.forward(parameters)

    assert energy_preserving_residual(entries) < 1e-12
    assert np.allclose(
        np.vdot(entries, entry_gradient),
        np.vdot(parameters, mapping.adjoint(entry_gradient)),
    )


@pytest.mark.parametrize("model_form", ["H", "AH", "cAH"])
def test_matrix_free_operator_has_correct_adjoint(model_form):
    generator = np.random.default_rng(12)
    states = generator.normal(size=(3, 30))
    ddts = generator.normal(size=states.shape)
    workspace = _EnergyPreservingFitWorkspace(states, ddts, model_form)
    operator, _, _ = workspace._matrix_free_operator(0.2)
    parameters = generator.normal(size=operator.shape[1])
    residual = generator.normal(size=operator.shape[0])

    assert np.allclose(
        np.vdot(operator @ parameters, residual),
        np.vdot(parameters, operator.rmatvec(residual)),
    )


def _model_entries(model, model_form):
    pieces = []
    if model_form == "cAH":
        pieces.append(model.c_.entries[:, None])
    if model_form in ("cAH", "AH"):
        pieces.append(model.A_.entries)
    pieces.append(model.H_.entries)
    return np.hstack(pieces)


@pytest.mark.parametrize("model_form", ["H", "AH", "cAH"])
def test_matrix_free_fit_matches_direct_global_fit(model_form):
    generator = np.random.default_rng(21)
    states = generator.normal(size=(3, 80))
    ddts = generator.normal(size=states.shape)

    direct = fit_energy_preserving_model(
        states, ddts, 0.2, model_form, solver="direct"
    )
    matrix_free = fit_energy_preserving_model(
        states,
        ddts,
        0.2,
        model_form,
        solver="matrix_free",
        solver_options={"atol": 1e-12, "btol": 1e-12},
    )

    assert np.allclose(
        _model_entries(matrix_free, model_form),
        _model_entries(direct, model_form),
        atol=1e-9,
        rtol=1e-9,
    )


def test_matrix_free_fit_does_not_assemble_kkt(monkeypatch):
    states, ddts, _ = _training_data(linear=np.eye(2))

    def fail(*args, **kwargs):
        del args, kwargs
        raise AssertionError("direct KKT assembly was called")

    monkeypatch.setattr(epopinf.sparse, "kron", fail)
    model = fit_energy_preserving_model(
        states, ddts, 0.1, solver="matrix_free"
    )

    assert energy_preserving_residual(model.H_.entries) < 1e-12


def test_verbose_fit_reports_problem_and_solver_diagnostics(capsys):
    states, ddts, _ = _training_data(linear=np.eye(2))

    fit_energy_preserving_model(
        states, ddts, 0.1, solver="matrix_free", verbose=1
    )

    output = capsys.readouterr().out
    assert "[EP-OpInf] prepared" in output
    assert "direct-hessian~" in output
    assert "selected=matrix_free" in output
    assert "[EP-OpInf] solved" in output
    assert "iterations=" in output
    assert "warm_start=False" in output


def test_auto_solver_switches_using_estimated_cost(monkeypatch):
    states, ddts, _ = _training_data(linear=np.eye(2))
    workspace = _EnergyPreservingFitWorkspace(states, ddts, "AH")

    assert workspace._resolve_solver("auto") == "direct"
    monkeypatch.setattr(epopinf, "_MATRIX_FREE_CSR_LIMIT", 1)
    assert workspace._resolve_solver("auto") == "matrix_free"


def test_matrix_free_nonconvergence_raises():
    generator = np.random.default_rng(17)
    states = generator.normal(size=(6, 120))
    ddts = generator.normal(size=states.shape)

    with pytest.raises(RuntimeError, match="failed to converge"):
        fit_energy_preserving_model(
            states,
            ddts,
            0.1,
            solver="matrix_free",
            solver_options={"maxiter": 1},
        )


@pytest.mark.parametrize("solver", ["bad", None])
def test_invalid_solver_raises(solver):
    states, ddts, _ = _training_data(linear=np.eye(2))
    with pytest.raises(ValueError, match="solver"):
        fit_energy_preserving_model(states, ddts, solver=solver)


def test_invalid_verbosity_raises():
    states, ddts, _ = _training_data(linear=np.eye(2))
    with pytest.raises(ValueError, match="verbose"):
        fit_energy_preserving_model(states, ddts, verbose=3)
