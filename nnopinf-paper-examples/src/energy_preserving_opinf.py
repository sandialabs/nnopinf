"""Energy-preserving operator inference for linear--quadratic models."""

import math
import time
import warnings

import numpy as np
import opinf
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg


_MATRIX_FREE_CSR_LIMIT = 64 * 2**20
_MATRIX_FREE_KKT_LIMIT = 20_000
_LSMR_DEFAULTS = {
    "atol": 1e-8,
    "btol": 1e-8,
    "conlim": 1e8,
    "maxiter": 2_000,
}
_LSMR_SUCCESS_CODES = {0, 1, 2, 4, 5}


def _validate_model_form(model_form):
    if model_form not in ("cAH", "AH", "H"):
        raise ValueError("model_form must be 'cAH', 'AH', or 'H'")
    return model_form


def energy_preserving_constraint_matrix(state_dimension, model_form="AH"):
    """Return the constraint matrix for a ``cAH``, ``AH``, or ``H`` EP model.

    ``H`` uses the compressed quadratic ordering of
    :class:`opinf.operators.QuadraticOperator`.
    """
    model_form = _validate_model_form(model_form)
    r = int(state_dimension)
    if r < 1:
        raise ValueError("state_dimension must be positive")

    quadratic_pairs = opinf.operators.QuadraticOperator.ckron_indices(r)
    pair_to_column = {
        tuple(sorted((int(i), int(j)))): column
        for column, (i, j) in enumerate(quadratic_pairs)
    }
    quadratic_dimension = len(quadratic_pairs)
    quadratic_offset = {"cAH": r + 1, "AH": r, "H": 0}[model_form]
    operator_dimension = quadratic_offset + quadratic_dimension

    rows = []
    columns = []
    values = []
    constraint = 0
    for i in range(r):
        for j in range(i + 1):
            for k in range(j + 1):
                terms = (
                    (i, j, k, 1.0 if j == k else 0.5),
                    (j, i, k, 1.0 if i == k else 0.5),
                    (k, j, i, 1.0 if j == i else 0.5),
                )
                for output, first, second, coefficient in terms:
                    rows.append(constraint)
                    columns.append(
                        output * operator_dimension
                        + quadratic_offset
                        + pair_to_column[tuple(sorted((first, second)))]
                    )
                    values.append(coefficient)
                constraint += 1

    shape = (math.comb(r + 2, 3), r * operator_dimension)
    return sparse.coo_matrix((values, (rows, columns)), shape=shape).tocsr()


def energy_preserving_residual(quadratic_entries):
    """Return the maximum absolute EP constraint residual for ``H``."""
    H = np.asarray(quadratic_entries)
    if H.ndim != 2:
        raise ValueError("quadratic_entries must be a matrix")
    r = H.shape[0]
    quadratic_dimension = r * (r + 1) // 2
    if H.shape[1] == r**2:
        H = opinf.operators.QuadraticOperator.compress_entries(H)
    if H.shape != (r, quadratic_dimension):
        raise ValueError("invalid quadratic operator dimensions")

    quadratic_pairs = opinf.operators.QuadraticOperator.ckron_indices(r)
    pair_to_column = {
        tuple(sorted((int(i), int(j)))): column
        for column, (i, j) in enumerate(quadratic_pairs)
    }
    residual = 0.0
    for i in range(r):
        for j in range(i + 1):
            for k in range(j + 1):
                terms = (
                    (i, j, k, 1.0 if j == k else 0.5),
                    (j, i, k, 1.0 if i == k else 0.5),
                    (k, j, i, 1.0 if j == i else 0.5),
                )
                value = sum(
                    coefficient
                    * H[output, pair_to_column[tuple(sorted((first, second)))]]
                    for output, first, second, coefficient in terms
                )
                residual = max(residual, abs(float(value)))
    return residual


def _validate_solver(solver, solver_options):
    if solver not in ("auto", "direct", "matrix_free"):
        raise ValueError("solver must be 'auto', 'direct', or 'matrix_free'")
    if solver_options is None:
        return dict(_LSMR_DEFAULTS)
    if not isinstance(solver_options, dict):
        raise TypeError("solver_options must be a dictionary or None")
    unknown = set(solver_options) - set(_LSMR_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown solver options: {sorted(unknown)}")
    options = dict(_LSMR_DEFAULTS)
    options.update(solver_options)
    for name in ("atol", "btol", "conlim"):
        value = float(options[name])
        if value <= 0 or not np.isfinite(value):
            raise ValueError(f"solver option {name!r} must be positive and finite")
        options[name] = value
    maxiter = int(options["maxiter"])
    if maxiter < 1:
        raise ValueError("solver option 'maxiter' must be positive")
    options["maxiter"] = maxiter
    return options


def _validate_verbosity(verbose):
    if isinstance(verbose, (bool, np.bool_)):
        return int(verbose)
    try:
        verbose = int(verbose)
    except (TypeError, ValueError) as exc:
        raise ValueError("verbose must be 0, 1, or 2") from exc
    if verbose not in (0, 1, 2):
        raise ValueError("verbose must be 0, 1, or 2")
    return verbose


class _SkewQuadraticMap:
    """Map skew-block coefficients to compressed quadratic entries."""

    def __init__(self, state_dimension):
        self.state_dimension = int(state_dimension)
        self.row_indices, self.column_indices = np.triu_indices(
            self.state_dimension, 1
        )
        self.block_dimension = len(self.row_indices)
        pairs = opinf.operators.QuadraticOperator.ckron_indices(
            self.state_dimension
        )
        pair_to_column = {
            tuple(sorted((int(i), int(j)))): column
            for column, (i, j) in enumerate(pairs)
        }
        self.diagonal_columns = np.array(
            [pair_to_column[(i, i)] for i in range(self.state_dimension)]
        )
        off_first = []
        off_second = []
        off_columns = []
        for first in range(self.state_dimension):
            for second in range(first):
                off_first.append(first)
                off_second.append(second)
                off_columns.append(pair_to_column[(second, first)])
        self.off_first = np.asarray(off_first, dtype=int)
        self.off_second = np.asarray(off_second, dtype=int)
        self.off_columns = np.asarray(off_columns, dtype=int)
        self.parameter_first_columns = np.empty(
            (self.state_dimension, self.block_dimension), dtype=int
        )
        self.parameter_second_columns = np.empty_like(
            self.parameter_first_columns
        )
        for block in range(self.state_dimension):
            for pair, (first, second) in enumerate(
                zip(self.row_indices, self.column_indices)
            ):
                self.parameter_first_columns[block, pair] = pair_to_column[
                    tuple(sorted((block, second)))
                ]
                self.parameter_second_columns[block, pair] = pair_to_column[
                    tuple(sorted((block, first)))
                ]

    @property
    def parameter_dimension(self):
        return self.state_dimension * self.block_dimension

    def forward(self, parameters):
        r = self.state_dimension
        values = np.asarray(parameters).reshape(r, self.block_dimension)
        blocks = np.zeros((r, r, r), dtype=values.dtype)
        blocks[:, self.row_indices, self.column_indices] = values
        blocks[:, self.column_indices, self.row_indices] = -values

        entries = np.empty((r, r * (r + 1) // 2), dtype=values.dtype)
        entries[:, self.diagonal_columns] = blocks[
            np.arange(r), :, np.arange(r)
        ].T
        entries[:, self.off_columns] = (
            blocks[self.off_first, :, self.off_second].T
            + blocks[self.off_second, :, self.off_first].T
        )
        return entries

    def adjoint(self, entry_gradient):
        r = self.state_dimension
        gradient = np.asarray(entry_gradient)
        blocks = np.zeros((r, r, r), dtype=gradient.dtype)
        blocks[np.arange(r), :, np.arange(r)] = gradient[
            :, self.diagonal_columns
        ].T
        blocks[self.off_first, :, self.off_second] = gradient[
            :, self.off_columns
        ].T
        blocks[self.off_second, :, self.off_first] = gradient[
            :, self.off_columns
        ].T
        values = (
            blocks[:, self.row_indices, self.column_indices]
            - blocks[:, self.column_indices, self.row_indices]
        )
        return values.ravel()

    def column_norms_squared(self, quadratic_feature_norms_squared):
        feature_norms = np.asarray(quadratic_feature_norms_squared)
        return (
            feature_norms[self.parameter_first_columns]
            + feature_norms[self.parameter_second_columns]
        ).ravel()


class _EnergyPreservingFitWorkspace:
    """Reusable data and solvers for one EP-OpInf training dataset."""

    def __init__(self, states, ddts, model_form="AH"):
        preparation_start = time.perf_counter()
        self.model_form = _validate_model_form(model_form)
        self.states = np.asarray(states, dtype=float)
        self.ddts = np.asarray(ddts, dtype=float)
        if self.states.ndim != 2 or self.ddts.shape != self.states.shape:
            raise ValueError(
                "states and ddts must have the same two-dimensional shape"
            )
        if not np.all(np.isfinite(self.states)) or not np.all(
            np.isfinite(self.ddts)
        ):
            raise ValueError("states and ddts must contain only finite values")

        self.state_dimension = self.states.shape[0]
        quadratic_data = opinf.operators.QuadraticOperator.datablock(self.states)
        if self.model_form == "cAH":
            self.data_matrix = np.hstack(
                (
                    np.ones((self.states.shape[1], 1)),
                    self.states.T,
                    quadratic_data.T,
                )
            )
        elif self.model_form == "AH":
            self.data_matrix = np.hstack((self.states.T, quadratic_data.T))
        else:
            self.data_matrix = quadratic_data.T
        self.operator_dimension = self.data_matrix.shape[1]
        self.quadratic_dimension = quadratic_data.shape[0]
        self.quadratic_offset = self.operator_dimension - self.quadratic_dimension
        self.skew_map = _SkewQuadraticMap(self.state_dimension)
        self.prefix_parameter_dimension = (
            self.state_dimension * self.quadratic_offset
        )
        self.parameter_dimension = (
            self.prefix_parameter_dimension + self.skew_map.parameter_dimension
        )
        feature_norms_squared = np.sum(self.data_matrix**2, axis=0)
        parameter_norms_squared = []
        regularizer_weights = []
        if self.prefix_parameter_dimension:
            parameter_norms_squared.append(
                np.tile(
                    feature_norms_squared[: self.quadratic_offset],
                    self.state_dimension,
                )
            )
            regularizer_weights.append(
                np.ones(self.prefix_parameter_dimension)
            )
        parameter_norms_squared.append(
            self.skew_map.column_norms_squared(
                feature_norms_squared[self.quadratic_offset :]
            )
        )
        regularizer_weights.append(
            np.full(self.skew_map.parameter_dimension, 2.0)
        )
        self._parameter_norms_squared = np.concatenate(
            parameter_norms_squared
        )
        self._regularizer_weights = np.concatenate(regularizer_weights)
        self.preparation_seconds = time.perf_counter() - preparation_start
        self._reported_problem_size = False

    def cost_estimate(self):
        """Return direct-solver size estimates used by automatic dispatch."""
        r = self.state_dimension
        p = self.operator_dimension
        hessian_rows = r * p
        return {
            "hessian_csr_bytes": r * p * p * 12 + (hessian_rows + 1) * 4,
            "kkt_dimension": hessian_rows + math.comb(r + 2, 3),
            "design_bytes": self.data_matrix.nbytes,
        }

    def _resolve_solver(self, solver):
        if solver != "auto":
            return solver
        r = self.state_dimension
        p = self.operator_dimension
        estimates = self.cost_estimate()
        if (
            estimates["hessian_csr_bytes"] >= _MATRIX_FREE_CSR_LIMIT
            or estimates["kkt_dimension"] >= _MATRIX_FREE_KKT_LIMIT
        ):
            return "matrix_free"
        return "direct"

    def _parameters_to_entries(self, parameters):
        r = self.state_dimension
        prefix_end = self.prefix_parameter_dimension
        entries = np.empty((r, self.operator_dimension), dtype=float)
        if prefix_end:
            entries[:, : self.quadratic_offset] = parameters[
                :prefix_end
            ].reshape(r, self.quadratic_offset)
        entries[:, self.quadratic_offset :] = self.skew_map.forward(
            parameters[prefix_end:]
        )
        return entries

    def _entries_to_parameter_gradient(self, entry_gradient):
        pieces = []
        if self.prefix_parameter_dimension:
            pieces.append(
                entry_gradient[:, : self.quadratic_offset].ravel()
            )
        pieces.append(
            self.skew_map.adjoint(entry_gradient[:, self.quadratic_offset :])
        )
        return np.concatenate(pieces)

    def _matrix_free_operator(self, regularizer):
        r = self.state_dimension
        m = self.states.shape[1]
        p = self.operator_dimension
        data_rows = r * m
        coefficient_rows = r * p

        norms = (
            self._parameter_norms_squared
            + regularizer**2 * self._regularizer_weights
        )
        scales = np.ones_like(norms)
        nonzero = norms > 0
        scales[nonzero] = 1.0 / np.sqrt(norms[nonzero])

        def matvec(scaled_parameters):
            parameters = scales * np.asarray(scaled_parameters)
            entries = self._parameters_to_entries(parameters)
            prediction = entries @ self.data_matrix.T
            return np.concatenate(
                (prediction.ravel(), regularizer * entries.ravel())
            )

        def rmatvec(residual):
            residual = np.asarray(residual)
            data_residual = residual[:data_rows].reshape(r, m)
            coefficient_residual = residual[data_rows:].reshape(r, p)
            entry_gradient = data_residual @ self.data_matrix
            entry_gradient += regularizer * coefficient_residual
            return scales * self._entries_to_parameter_gradient(entry_gradient)

        operator = sparse_linalg.LinearOperator(
            (data_rows + coefficient_rows, self.parameter_dimension),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=float,
        )
        rhs = np.concatenate((self.ddts.ravel(), np.zeros(coefficient_rows)))
        return operator, rhs, scales

    def _solve_matrix_free(self, regularizer, options, initial_guess, verbose):
        operator, rhs, scales = self._matrix_free_operator(regularizer)
        scaled_guess = None
        if initial_guess is not None:
            initial_guess = np.asarray(initial_guess, dtype=float)
            if initial_guess.shape != (self.parameter_dimension,):
                raise ValueError("invalid matrix-free initial guess dimensions")
            scaled_guess = initial_guess / scales
        result = sparse_linalg.lsmr(
            operator,
            rhs,
            x0=scaled_guess,
            show=verbose >= 2,
            **options,
        )
        scaled_solution = result[0]
        istop, iterations, normr, normar, norma, conda, normx = result[1:]
        self.last_solver_info = {
            "solver": "matrix_free",
            "istop": istop,
            "iterations": iterations,
            "normr": normr,
            "normar": normar,
            "norma": norma,
            "conda": conda,
            "normx": normx,
        }
        if istop not in _LSMR_SUCCESS_CODES:
            raise RuntimeError(
                "EP-OpInf matrix-free solve failed to converge "
                f"(istop={istop}, iterations={iterations}, "
                f"normr={normr:.3e}, normar={normar:.3e}, "
                f"norma={norma:.3e}, conda={conda:.3e}, "
                f"normx={normx:.3e})"
            )
        parameters = scales * scaled_solution
        return self._parameters_to_entries(parameters), parameters

    def _solve_direct(self, regularizer):
        r = self.state_dimension
        p = self.operator_dimension
        gramian = self.data_matrix.T @ self.data_matrix
        if regularizer:
            gramian = gramian + regularizer**2 * np.eye(p)
        hessian = sparse.kron(
            sparse.eye(r), sparse.csr_matrix(gramian), format="csr"
        )
        constraints = energy_preserving_constraint_matrix(r, self.model_form)
        kkt = sparse.bmat(
            [[hessian, constraints.T], [constraints, None]], format="csc"
        )
        rhs = np.concatenate(
            ((self.ddts @ self.data_matrix).ravel(), np.zeros(constraints.shape[0]))
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", sparse_linalg.MatrixRankWarning)
            try:
                solution = sparse_linalg.spsolve(kkt, rhs)
            except (sparse_linalg.MatrixRankWarning, RuntimeError) as exc:
                raise RuntimeError("EP-OpInf constrained solve failed") from exc
        self.last_solver_info = {"solver": "direct"}
        return solution[: r * p].reshape(r, p), None

    def fit_model(
        self,
        regularizer=0.0,
        *,
        solver="auto",
        solver_options=None,
        initial_guess=None,
        verbose=0,
    ):
        verbose = _validate_verbosity(verbose)
        regularizer = float(regularizer)
        if regularizer < 0 or not np.isfinite(regularizer):
            raise ValueError(
                "regularizer must be a finite nonnegative scalar"
            )
        options = _validate_solver(solver, solver_options)
        resolved_solver = self._resolve_solver(solver)
        if verbose and not self._reported_problem_size:
            estimates = self.cost_estimate()
            print(
                "[EP-OpInf] prepared "
                f"form={self.model_form} r={self.state_dimension} "
                f"snapshots={self.states.shape[1]} "
                f"features={self.operator_dimension} "
                f"parameters={self.parameter_dimension} "
                f"prep={self.preparation_seconds:.3f}s "
                f"design={estimates['design_bytes'] / 2**20:.1f}MiB "
                f"direct-hessian~{estimates['hessian_csr_bytes'] / 2**20:.1f}MiB "
                f"kkt-dimension={estimates['kkt_dimension']} "
                f"selected={resolved_solver}"
            )
            self._reported_problem_size = True
        solve_start = time.perf_counter()
        if resolved_solver == "matrix_free":
            entries, parameters = self._solve_matrix_free(
                regularizer, options, initial_guess, verbose
            )
        else:
            entries, parameters = self._solve_direct(regularizer)
        solve_seconds = time.perf_counter() - solve_start
        if not np.all(np.isfinite(entries)):
            raise RuntimeError(
                "EP-OpInf constrained solve produced non-finite entries"
            )
        H = entries[:, self.quadratic_offset :]
        residual = energy_preserving_residual(H)
        tolerance = 1e-9 * max(1.0, float(np.linalg.norm(H, ord=np.inf)))
        if residual > tolerance:
            raise RuntimeError(
                f"EP-OpInf constraint residual {residual:.3e} "
                f"exceeds {tolerance:.3e}"
            )
        if verbose:
            details = ""
            if resolved_solver == "matrix_free":
                details = (
                    f" iterations={self.last_solver_info['iterations']}"
                    f" istop={self.last_solver_info['istop']}"
                    f" normr={self.last_solver_info['normr']:.3e}"
                    f" normar={self.last_solver_info['normar']:.3e}"
                    f" conda={self.last_solver_info['conda']:.3e}"
                    f" warm_start={initial_guess is not None}"
                )
            print(
                "[EP-OpInf] solved "
                f"lambda={regularizer:.3e} backend={resolved_solver} "
                f"time={solve_seconds:.3f}s residual={residual:.3e}{details}"
            )
        return _model_from_entries(entries, self.model_form), parameters


def _model_from_entries(entries, model_form):
    r = entries.shape[0]
    quadratic_offset = {"cAH": r + 1, "AH": r, "H": 0}[model_form]
    H = entries[:, quadratic_offset:]
    model_operators = [opinf.operators.QuadraticOperator(H)]
    if model_form in ("cAH", "AH"):
        linear_offset = 1 if model_form == "cAH" else 0
        model_operators.insert(
            0,
            opinf.operators.LinearOperator(
                entries[:, linear_offset : linear_offset + r]
            ),
        )
    if model_form == "cAH":
        model_operators.insert(0, opinf.operators.ConstantOperator(entries[:, 0]))
    return opinf.models.ContinuousModel(model_operators)


def fit_energy_preserving_model(
    states,
    ddts,
    regularizer=0.0,
    model_form="AH",
    *,
    solver="auto",
    solver_options=None,
    verbose=0,
):
    """Fit a ``cAH``, ``AH``, or ``H`` model with a hard EP constraint.

    ``solver="auto"`` uses the direct KKT method for small systems and a
    skew-parameterized matrix-free least-squares method for larger systems.
    The matrix-free method minimizes the same regularized global objective as
    the direct method while satisfying the EP constraint by construction.
    """
    workspace = _EnergyPreservingFitWorkspace(states, ddts, model_form)
    model, _ = workspace.fit_model(
        regularizer,
        solver=solver,
        solver_options=solver_options,
        verbose=verbose,
    )
    return model


def interpolate_energy_preserving_models(parameters, models, model_form="AH"):
    """Interpolate populated EP models with the standard ``opinf`` policy."""
    model_form = _validate_model_form(model_form)
    parameters = np.asarray(parameters)
    if len(models) != len(parameters):
        raise ValueError("one model is required for each parameter value")
    if len(models) < 2:
        raise ValueError("at least two models are required for interpolation")

    quadratic_entries = [model.H_.entries for model in models]
    model_operators = [
        opinf.operators.InterpQuadraticOperator(parameters, quadratic_entries)
    ]
    if model_form in ("cAH", "AH"):
        linear_entries = [model.A_.entries for model in models]
        model_operators.insert(
            0,
            opinf.operators.InterpLinearOperator(parameters, linear_entries),
        )
    if model_form == "cAH":
        constant_entries = [model.c_.entries for model in models]
        model_operators.insert(
            0,
            opinf.operators.InterpConstantOperator(parameters, constant_entries),
        )
    return opinf.models.InterpolatedContinuousModel(
        model_operators
    )
