"""Time integration utilities for ``opinf`` continuous models."""

import numpy as np
import scipy.optimize


_FIRST_ORDER_INTEGRATORS = {"rk4", "crank-nicolson", "implicit-midpoint"}
_SPECIAL_MODEL_INTEGRATORS = {
    "LOpInf": "newmark",
    "LOpInf-SpML": "newmark",
    "entropy": "rk4",
}


def _normalize_integrator(integration):
    if not isinstance(integration, str):
        raise ValueError("integrator values must be strings")
    integration = integration.strip().lower()
    if integration == "cn":
        return "crank-nicolson"
    return integration


def get_model_integrator(input_yaml, model_type):
    """Resolve the runtime integrator for an exact model-type identifier."""
    mapping = input_yaml.get("model-integrator", {})
    if not isinstance(mapping, dict):
        raise ValueError("model-integrator must be a mapping")
    if model_type in mapping:
        integration = _normalize_integrator(mapping[model_type])
    elif model_type in _SPECIAL_MODEL_INTEGRATORS:
        integration = _SPECIAL_MODEL_INTEGRATORS[model_type]
    else:
        fom_settings = input_yaml.get("fom", {})
        if not isinstance(fom_settings, dict):
            raise ValueError("fom must be a mapping")
        integration = _normalize_integrator(fom_settings.get("integration", "rk4"))

    if model_type in ("LOpInf", "LOpInf-SpML"):
        if integration != "newmark":
            raise ValueError(f"{model_type} only supports the 'newmark' integrator")
    elif model_type == "entropy":
        if integration != "rk4":
            raise ValueError("entropy only supports the 'rk4' integrator")
    elif integration not in _FIRST_ORDER_INTEGRATORS:
        raise ValueError(
            f"unsupported integrator {integration!r} for model {model_type!r}"
        )
    return integration


def _model_rhs(model, state, parameter):
    if parameter is None:
        return np.asarray(model.rhs(0.0, state)).reshape(-1)
    return np.asarray(model.rhs(0.0, parameter, state)).reshape(-1)


def _model_jacobian(model, state, parameter):
    if parameter is None:
        return np.asarray(model.jacobian(0.0, state))
    return np.asarray(model.jacobian(0.0, parameter, state))


def first_order_step(
    rhs,
    state,
    dt,
    integration="rk4",
    jacobian=None,
    tolerance=1e-10,
):
    """Advance an autonomous first-order system by one configured step."""
    integration = _normalize_integrator(integration)
    if integration not in _FIRST_ORDER_INTEGRATORS:
        raise ValueError(
            "integration must be 'rk4', 'crank-nicolson'/'cn', "
            "or 'implicit-midpoint'"
        )
    state = np.asarray(state, dtype=float).reshape(-1)
    dt = float(dt)
    tolerance = float(tolerance)
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("dt must be a finite positive scalar")
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be a finite positive scalar")
    if not np.all(np.isfinite(state)):
        raise ValueError("state must contain only finite values")

    if integration == "rk4":
        next_state = state.copy()
        for coefficient in (1.0 / 4.0, 1.0 / 3.0, 1.0 / 2.0, 1.0):
            next_state = state + coefficient * dt * np.asarray(
                rhs(next_state)
            ).reshape(-1)
        return next_state

    identity = np.eye(state.size)
    if integration == "crank-nicolson":
        initial_rhs = np.asarray(rhs(state)).reshape(-1)

        def residual(next_state):
            return next_state - state - 0.5 * dt * (
                initial_rhs + np.asarray(rhs(next_state)).reshape(-1)
            )

        def residual_jacobian(next_state):
            return identity - 0.5 * dt * np.asarray(jacobian(next_state))

    else:

        def residual(next_state):
            midpoint = 0.5 * (state + next_state)
            return next_state - state - dt * np.asarray(rhs(midpoint)).reshape(-1)

        def residual_jacobian(next_state):
            midpoint = 0.5 * (state + next_state)
            return identity - 0.5 * dt * np.asarray(jacobian(midpoint))

    root_kwargs = {}
    if jacobian is not None:
        root_kwargs["jac"] = residual_jacobian
    result = scipy.optimize.root(
        residual,
        state,
        method="hybr",
        options={"xtol": tolerance, "maxfev": 100 * (state.size + 1)},
        **root_kwargs,
    )
    residual_norm = float(np.linalg.norm(residual(result.x)))
    residual_tolerance = tolerance * max(1.0, float(np.linalg.norm(state)))
    if (
        not np.all(np.isfinite(result.x))
        or not np.isfinite(residual_norm)
        or residual_norm > residual_tolerance
    ):
        raise RuntimeError(
            f"{integration} solve failed: {result.message}; "
            f"residual={residual_norm:.3e}, "
            f"tolerance={residual_tolerance:.3e}"
        )
    return np.asarray(result.x)


def _implicit_midpoint_system(model, state, dt, parameter):
    identity = np.eye(state.size)

    def residual(next_state):
        midpoint = 0.5 * (state + next_state)
        return next_state - state - dt * _model_rhs(model, midpoint, parameter)

    def jacobian(next_state):
        midpoint = 0.5 * (state + next_state)
        return identity - 0.5 * dt * _model_jacobian(model, midpoint, parameter)

    return residual, jacobian


def _validate_times_and_state(initial_state, times):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must be a one-dimensional array with at least two entries")
    steps = np.diff(times)
    if np.any(steps <= 0) or not np.all(np.isfinite(steps)):
        raise ValueError("times must be finite and strictly increasing")
    state = np.asarray(initial_state, dtype=float).reshape(-1)
    if not np.all(np.isfinite(state)):
        raise ValueError("initial_state must contain only finite values")
    return times, steps, state


def rk4_predict(model, initial_state, times, parameter=None):
    """Integrate using the four-stage RK scheme used by the ROM drivers."""
    times, steps, state = _validate_times_and_state(initial_state, times)
    states = np.empty((state.size, times.size), dtype=float)
    states[:, 0] = state
    for index, dt in enumerate(steps, start=1):
        state = first_order_step(
            lambda value: _model_rhs(model, value, parameter),
            state,
            dt,
            integration="rk4",
        )
        states[:, index] = state
    return states


def crank_nicolson_step(model, state, dt, parameter=None, tolerance=1e-10):
    """Advance a continuous ``opinf`` model by one Crank--Nicolson step."""
    return first_order_step(
        lambda value: _model_rhs(model, value, parameter),
        state,
        dt,
        integration="crank-nicolson",
        jacobian=lambda value: _model_jacobian(model, value, parameter),
        tolerance=tolerance,
    )


def crank_nicolson_predict(model, initial_state, times, parameter=None):
    """Integrate a continuous model with Crank--Nicolson."""
    times, steps, state = _validate_times_and_state(initial_state, times)
    states = np.empty((state.size, times.size), dtype=float)
    states[:, 0] = state
    for index, dt in enumerate(steps, start=1):
        try:
            state = crank_nicolson_step(model, state, dt, parameter=parameter)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Crank--Nicolson solve failed at step {index}, "
                f"t={times[index]:.16g}"
            ) from exc
        states[:, index] = state
    return states


def implicit_midpoint_step(model, state, dt, parameter=None, tolerance=1e-10):
    """Advance a continuous ``opinf`` model by one implicit-midpoint step."""
    return first_order_step(
        lambda value: _model_rhs(model, value, parameter),
        state,
        dt,
        integration="implicit-midpoint",
        jacobian=lambda value: _model_jacobian(model, value, parameter),
        tolerance=tolerance,
    )


def implicit_midpoint_predict(model, initial_state, times, parameter=None):
    """Integrate a continuous ``opinf`` model at the requested time points."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must be a one-dimensional array with at least two entries")
    steps = np.diff(times)
    if np.any(steps <= 0) or not np.all(np.isfinite(steps)):
        raise ValueError("times must be finite and strictly increasing")

    state = np.asarray(initial_state, dtype=float).reshape(-1)
    states = np.empty((state.size, times.size), dtype=float)
    states[:, 0] = state
    for index, dt in enumerate(steps, start=1):
        try:
            state = implicit_midpoint_step(model, state, dt, parameter=parameter)
        except RuntimeError as exc:
            raise RuntimeError(
                f"implicit midpoint solve failed at step {index}, "
                f"t={times[index]:.16g}"
            ) from exc
        states[:, index] = state
    return states


def predict_continuous_model(
    model, initial_state, times, parameter=None, integration="rk4"
):
    """Integrate with a supported YAML integration method.

    Parameters
    ----------
    integration : {"rk4", "crank-nicolson", "cn", "implicit-midpoint"}
        Time integration method. Defaults to ``"rk4"``.
    """
    integration = str(integration).strip().lower()
    if integration == "rk4":
        return rk4_predict(model, initial_state, times, parameter=parameter)
    if integration in ("crank-nicolson", "cn"):
        return crank_nicolson_predict(
            model, initial_state, times, parameter=parameter
        )
    if integration == "implicit-midpoint":
        return implicit_midpoint_predict(
            model, initial_state, times, parameter=parameter
        )
    raise ValueError(
        "integration must be 'rk4', 'crank-nicolson', 'cn', "
        "or 'implicit-midpoint'"
    )
