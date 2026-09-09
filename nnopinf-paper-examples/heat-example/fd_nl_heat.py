#!/usr/bin/env python3
"""
Finite difference solver for 2D transient nonlinear heat conduction on [0,1]^2.
Dirichlet BCs are weakly enforced with a penalty term added to the RHS and matrix
at boundary nodes.
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


Array = np.ndarray


def kappa(T: float, params: Array | None = None) -> float:
    k0 = 1e-2
    w = 2e-2
    if params is None or len(params) < 2:
        k1 = 0.2
        Tc = 0.3
    else:
        k1, Tc = params[:4]
    kappa = 0.5*(k0 + k1) + 0.5*(k1 - k0) * np.tanh((T - Tc)/w)
    return kappa


def g_dirichlet(x: float, y: float, t: float) -> float:
    return 0.0


def source_f(x: float, y: float, t: float) -> float:
    return 1.0


def initial_u(x: float, y: float) -> float:
    return 0.0*math.sin(math.pi * x) * math.sin(math.pi * y)


def build_operator(
    nx: int,
    ny: int,
    u: Array,
    k_func: Callable[[Array, Array | None], Array],
    gamma: float,
    t: float,
    g_func: Callable[[float, float, float], float],
    X: Array,
    Y: Array,
    bnd_mask: Array,
    params: Array | None = None,
) -> Tuple[sp.csr_matrix, Array]:
    """Assemble diffusion operator and penalty RHS for weak Dirichlet BCs."""
    hx = 1.0 / nx
    hy = 1.0 / ny
    n = (nx + 1) * (ny + 1)
    rhs_bnd = np.zeros(n, dtype=float)

    u_grid = u.reshape((ny + 1, nx + 1))
    try:
        k_grid = k_func(u_grid, params)
    except TypeError:
        k_grid = k_func(u_grid)

    # Harmonic averages on faces.
    ke = 2.0 * k_grid[:, :-1] * k_grid[:, 1:] / (k_grid[:, :-1] + k_grid[:, 1:])
    kn = 2.0 * k_grid[:-1, :] * k_grid[1:, :] / (k_grid[:-1, :] + k_grid[1:, :])

    ae = np.zeros_like(k_grid)
    aw = np.zeros_like(k_grid)
    an = np.zeros_like(k_grid)
    a_s = np.zeros_like(k_grid)
    hx2 = hx * hx
    hy2 = hy * hy
    ae[1:ny, 1:nx] = ke[1:ny, 1:nx] / hx2
    aw[1:ny, 1:nx] = ke[1:ny, 0:nx - 1] / hx2
    an[1:ny, 1:nx] = kn[1:ny, 1:nx] / hy2
    a_s[1:ny, 1:nx] = kn[0:ny - 1, 1:nx] / hy2

    ap = ae + aw + an + a_s

    diag0 = ap.ravel()
    diagE = (-ae.ravel())[:-1]
    diagW = (-aw.ravel())[1:]
    diagN = (-an.ravel())[: n - (nx + 1)]
    diagS = (-a_s.ravel())[(nx + 1):]

    # Weak Dirichlet penalty at boundary nodes.
    h = min(hx, hy)
    penalty = gamma * k_grid / h
    diag0 = diag0 + (penalty * bnd_mask).ravel()

    gvals = np.vectorize(g_func)(X, Y, t)
    rhs_bnd = (penalty * bnd_mask * gvals).ravel()

    A = sp.diags(
        diagonals=[diag0, diagE, diagW, diagN, diagS],
        offsets=[0, 1, -1, nx + 1, -(nx + 1)],
        shape=(n, n),
        format="csr",
    )
    return A, rhs_bnd


def assemble_source(
    t: float,
    f_func: Callable[[float, float, float], float],
    X: Array,
    Y: Array,
) -> Array:
    fvals = np.vectorize(f_func)(X, Y, t)
    return fvals.ravel().astype(float)


class heat_fom_fd2d:
    def __init__(
        self,
        nx: int,
        ny: int,
        dt: float,
        t_end: float,
        k_func: Callable[[Array, Array | None], Array],
        g_func: Callable[[float, float, float], float],
        f_func: Callable[[float, float, float], float],
        u0_func: Callable[[float, float], float],
        gamma: float = 20.0,
        max_iter: int = 30,
        tol: float = 1.0e-8,
        time_integrator: str = "cn",
    ) -> None:
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.t_end = t_end
        self.k_func = k_func
        self.g_func = g_func
        self.f_func = f_func
        self.u0_func = u0_func
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.time_integrator = time_integrator.lower()
        self.params: Array | None = None

        self.hx = 1.0 / nx
        self.hy = 1.0 / ny
        self.n = (nx + 1) * (ny + 1)

        xs = np.linspace(0.0, 1.0, nx + 1)
        ys = np.linspace(0.0, 1.0, ny + 1)
        self.X, self.Y = np.meshgrid(xs, ys)
        self.bnd_mask = np.zeros((ny + 1, nx + 1), dtype=float)
        self.bnd_mask[0, :] = 1.0
        self.bnd_mask[-1, :] = 1.0
        self.bnd_mask[:, 0] = 1.0
        self.bnd_mask[:, -1] = 1.0

        u0 = np.vectorize(u0_func)(self.X, self.Y)
        self.u0 = u0.ravel().astype(float)

    def build_operator(self, u: Array, t: float) -> Tuple[sp.csr_matrix, Array]:
        return build_operator(
            self.nx,
            self.ny,
            u,
            self.k_func,
            self.gamma,
            t,
            self.g_func,
            self.X,
            self.Y,
            self.bnd_mask,
            params=self.params,
        )

    def velocity(self, u: Array, t: float) -> Array:
        A, rhs_bnd = self.build_operator(u, t)
        rhs_source = assemble_source(t, self.f_func, self.X, self.Y)
        return -A.dot(u) + rhs_source + rhs_bnd

    def solve_transient(self) -> Tuple[Array, Array]:
        self.params = None
        u = self.u0.copy()
        times = [0.0]
        sol = [u.copy()]
        nsteps = int(math.ceil(self.t_end / self.dt))

        if self.time_integrator == "rk4":
            rk4const = np.array([1.0 / 4.0, 1.0 / 3.0, 1.0 / 2.0, 1.0], dtype=float)
            for step in range(1, nsteps + 1):
                t_n = (step - 1) * self.dt
                uhat0 = u.copy()
                uhat = u.copy()
                for i in range(4):
                    fhat = self.velocity(uhat, t_n)
                    uhat = uhat0 + self.dt * rk4const[i] * fhat
                u = uhat
                #if step % 10 == 0:
                #    print(f"rk4 step={step}")
                times.append(step * self.dt)
                sol.append(u.copy())
        else:
            I_dt = sp.eye(self.n, format="csr") / self.dt
            for step in range(1, nsteps + 1):
                t_n = (step - 1) * self.dt
                t_np1 = step * self.dt
                rhs_source_n = assemble_source(t_n, self.f_func, self.X, self.Y)
                rhs_source_np1 = assemble_source(t_np1, self.f_func, self.X, self.Y)
                A_n, rhs_bnd_n = self.build_operator(u, t_n)
                u_iter = u.copy()

                for it in range(self.max_iter):
                    A_np1, rhs_bnd_np1 = self.build_operator(u_iter, t_np1)
                    lhs = I_dt + 0.5 * A_np1
                    rhs = (I_dt - 0.5 * A_n).dot(u)
                    rhs += 0.5 * (rhs_source_np1 + rhs_source_n)
                    rhs += 0.5 * (rhs_bnd_np1 + rhs_bnd_n)
                    u_new = spla.spsolve(lhs, rhs)
                    err = np.linalg.norm(u_new - u_iter) / (np.linalg.norm(u_new) + 1.0e-14)
                    #print(f"step={step} iter={it} resid={err:.3e}")
                    u_iter = u_new
                    if err < self.tol:
                        break
                u = u_iter
                times.append(t_np1)
                sol.append(u.copy())

        return np.array(times), np.array(sol)

    def solve(self, params: Array, input_yaml: dict) -> Tuple[Array, Array, Array, Tuple[Array, Array], Array]:
        self.params = params
        dt = float(input_yaml["dt"])
        t_end = float(input_yaml["end-time"])
        snapshot_freq = int(input_yaml["snapshot-collect-frequency"])
        use_rk4 = self.time_integrator == "rk4"
        I_dt = None if use_rk4 else sp.eye(self.n, format="csr") / dt

        u = self.u0.copy()
        t = 0.0
        counter = 0
        t_history = []
        u_history = np.zeros((self.n, 0))
        f_history = np.zeros((self.n, 0))
        forcing = np.vectorize(self.f_func)(self.X, self.Y, 0.0).ravel().astype(float)

        rk4const = np.array([1.0 / 4.0, 1.0 / 3.0, 1.0 / 2.0, 1.0], dtype=float)
        while t <= t_end - dt / 2.0:
            if counter % snapshot_freq == 0:
                t_history.append(t)
                u_history = np.append(u_history, u[:, None], axis=1)
                f = np.vectorize(self.f_func)(self.X, self.Y, t).ravel().astype(float)
                f_history = np.append(f_history, f[:, None], axis=1)

            tn1 = t + dt
            if use_rk4:
                uhat0 = u.copy()
                uhat = u.copy()
                for i in range(4):
                    fhat = self.velocity(uhat, t)
                    uhat = uhat0 + dt * rk4const[i] * fhat
                u = uhat
                #if (counter + 1) % 10 == 0:
                #    print(f"rk4 step={counter + 1}")
            else:
                rhs_source_n = assemble_source(t, self.f_func, self.X, self.Y)
                rhs_source_np1 = assemble_source(tn1, self.f_func, self.X, self.Y)
                A_n, rhs_bnd_n = self.build_operator(u, t)
                u_iter = u.copy()

                for it in range(self.max_iter):
                    A_np1, rhs_bnd_np1 = self.build_operator(u_iter, tn1)
                    lhs = I_dt + 0.5 * A_np1
                    rhs = (I_dt - 0.5 * A_n).dot(u)
                    rhs += 0.5 * (rhs_source_np1 + rhs_source_n)
                    rhs += 0.5 * (rhs_bnd_np1 + rhs_bnd_n)
                    u_new = spla.spsolve(lhs, rhs)
                    err = np.linalg.norm(u_new - u_iter) / (np.linalg.norm(u_new) + 1.0e-14)
                    #print(f"step={counter + 1} iter={it} resid={err:.3e}")
                    u_iter = u_new
                    if err < self.tol:
                        break

                u = u_iter
            t = tn1
            counter += 1

        return u_history, np.array(t_history), forcing, (self.X, self.Y), f_history


def solve_transient(
    nx: int,
    ny: int,
    dt: float,
    t_end: float,
    k_func: Callable[[float], float],
    g_func: Callable[[float, float, float], float],
    f_func: Callable[[float, float, float], float],
    u0_func: Callable[[float, float], float],
    gamma: float = 20.0,
    max_iter: int = 30,
    tol: float = 1.0e-8,
    time_integrator: str = "cn",
) -> Tuple[Array, Array]:
    model = heat_fom_fd2d(
        nx,
        ny,
        dt,
        t_end,
        k_func,
        g_func,
        f_func,
        u0_func,
        gamma=gamma,
        max_iter=max_iter,
        tol=tol,
        time_integrator=time_integrator,
    )
    return model.solve_transient()


def main() -> None:
    import argparse
    import sys
    import yaml

    sys.path.append("../src/")
    from drivers import fom_driver

    parser = argparse.ArgumentParser()
    parser.add_argument("--i", help="Input yaml file", required=True)
    args = parser.parse_args()

    with open(args.i) as f:
        input_yaml_base = yaml.safe_load(f)

    fom_cfg = input_yaml_base.get("fom", {})
    nx = int(fom_cfg.get("nx", 40))
    ny = int(fom_cfg.get("ny", nx))
    dt = float(input_yaml_base.get("dt", 0.005))
    t_end = float(input_yaml_base.get("end-time", 0.2))
    gamma = float(fom_cfg.get("gamma", 50.0))
    time_integrator = str(input_yaml_base.get("integration", "cn"))

    model = heat_fom_fd2d(
        nx,
        ny,
        dt=dt,
        t_end=t_end,
        k_func=kappa,
        g_func=g_dirichlet,
        f_func=source_f,
        u0_func=initial_u,
        gamma=gamma,
        time_integrator=time_integrator,
    )
    fom_driver(model, input_yaml_base)


if __name__ == "__main__":
    main()
