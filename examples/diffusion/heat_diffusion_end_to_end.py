import argparse
import os

import numpy as np
import torch
from matplotlib import pyplot as plt

import nnopinf
import nnopinf.models as models
import nnopinf.training
import nnopinf.training.trainers


def laplacian_dirichlet(u, n, h):
    """Five-point Laplacian on an n x n interior grid with zero Dirichlet boundaries."""
    u2d = u.reshape(n, n)
    upad = np.pad(u2d, ((1, 1), (1, 1)), mode="constant")
    lap = (
        upad[2:, 1:-1]
        + upad[:-2, 1:-1]
        + upad[1:-1, 2:]
        + upad[1:-1, :-2]
        - 4.0 * upad[1:-1, 1:-1]
    ) / (h**2)
    return lap.reshape(-1)


class HeatDiffusionFOM:
    def __init__(self, n, forcing_value):
        self.n = n
        self.h = 1.0 / (n + 1)
        self.x = np.linspace(self.h, 1.0 - self.h, n)
        xx, yy = np.meshgrid(self.x, self.x, indexing="ij")
        self.u0 = (
            np.sin(np.pi * xx) * np.sin(np.pi * yy)
            + 0.25 * np.sin(2.0 * np.pi * xx) * np.sin(np.pi * yy)
        ).reshape(-1)
        self.forcing = forcing_value * np.ones(self.n * self.n)

    def velocity(self, u, kappa):
        return kappa * laplacian_dirichlet(u, self.n, self.h) + self.forcing

    def solve(self, dt, end_time, kappa):
        u = self.u0.copy()
        t = 0.0
        rk4const = np.array([1.0 / 4.0, 1.0 / 3.0, 1.0 / 2.0, 1.0])
        snapshots = []
        time = []
        while t <= end_time + 0.5 * dt:
            snapshots.append(u.copy())
            time.append(t)
            u0 = u.copy()
            for i in range(4):
                f = self.velocity(u, kappa)
                u = u0 + dt * rk4const[i] * f
            t += dt
        return np.array(snapshots).T, np.array(time)


class NnOpInfRom:
    def __init__(self, model):
        self.model_ = model
        self.inputs_ = {}

    def velocity(self, u):
        self.inputs_["x"] = torch.tensor(u[None], dtype=torch.float64)
        return self.model_.forward(self.inputs_)[0].detach().numpy()

    def solve(self, u0, dt, end_time):
        u = u0.copy()
        t = 0.0
        rk4const = np.array([1.0 / 4.0, 1.0 / 3.0, 1.0 / 2.0, 1.0])
        snapshots = []
        time = []
        while t <= end_time + 0.5 * dt:
            snapshots.append(u.copy())
            time.append(t)
            u0_l = u.copy()
            for i in range(4):
                f = self.velocity(u)
                u = u0_l + dt * rk4const[i] * f
            t += dt
        return np.array(snapshots).T, np.array(time)


def flatten_for_training(q):
    return np.reshape(q, (q.shape[0], q.shape[1] * q.shape[2])).T


def main():
    parser = argparse.ArgumentParser(
        description="Transient heat diffusion on the unit square with a linear SPD operator."
    )
    parser.add_argument("--grid-size", type=int, default=32, help="Interior points per direction.")
    parser.add_argument("--dt", type=float, default=2.0e-4, help="Time step for RK4.")
    parser.add_argument("--end-time", type=float, default=1.0, help="Final simulation time.")
    parser.add_argument("--rom-dim", type=int, default=20, help="Requested POD rank.")
    parser.add_argument("--kappa", type=float, default=0.75, help="Diffusion coefficient.")
    parser.add_argument("--forcing", type=float, default=1.0, help="Constant forcing amplitude.")
    parser.add_argument("--num-epochs", type=int, default=500, help="Training epochs.")
    parser.add_argument("--tr-delta0", type=float, default=1.0, help="Initial trust-region radius.")
    parser.add_argument("--tr-cg-max-iters", type=int, default=200, help="Maximum CG iterations per TR step.")
    parser.add_argument("--tr-batch-size", type=int, default=50, help="Batch size used by TR optimizer.")
    args = parser.parse_args()

    torch.manual_seed(1)
    np.random.seed(1)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(output_dir, "ml-models")
    os.makedirs(model_dir, exist_ok=True)

    fom = HeatDiffusionFOM(args.grid_size, args.forcing)
    u, _ = fom.solve(args.dt, args.end_time, args.kappa)
    snapshots_all = u[..., None]

    # POD basis from all training trajectories.
    snapshots_matrix = np.reshape(
        snapshots_all, (snapshots_all.shape[0], snapshots_all.shape[1] * snapshots_all.shape[2])
    )
    phi, singular_values, _ = np.linalg.svd(snapshots_matrix, full_matrices=False)
    relative_energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    pod_rank = np.searchsorted(relative_energy, 0.999999999) + 1
    rom_dim = min(args.rom_dim, int(pod_rank))
    phi = phi[:, :rom_dim]
    print(f"Using ROM dimension K={rom_dim}")

    # Reduced snapshots and time derivatives.
    uhat = np.einsum("ij,ikn->jkn", phi, snapshots_all)
    uhat_dot = (uhat[:, 2:, :] - uhat[:, :-2, :]) / (2.0 * args.dt)
    uhat = uhat[:, 1:-1, :]

    x_input = nnopinf.variables.Variable(size=rom_dim, name="x", normalization_strategy="MaxAbs")
    target = nnopinf.variables.Variable(size=rom_dim, name="y", normalization_strategy="MaxAbs")

    diffusion_operator = nnopinf.operators.LinearAffineSpdTensorOperator(
        acts_on=x_input, depends_on=(), positive=False
    )
    forcing_operator = nnopinf.operators.VectorOffsetOperator(n_outputs=rom_dim)
    model = models.Model([diffusion_operator, forcing_operator])

    training_settings = nnopinf.training.get_default_settings()
    training_settings["output-path"] = model_dir
    training_settings["optimizer"] = "ADAM"
    training_settings["batch-size"] = 5000
    training_settings["num-epochs"] = args.num_epochs
    training_settings["weight-decay"] = 1.0e-6
    training_settings["LBFGS-acceleration"] = True

    x_input.set_data(flatten_for_training(uhat))
    target.set_data(flatten_for_training(uhat_dot))

    print("Training linear SPD diffusion model")
    nnopinf.training.trainers.train(
        model, variables=[x_input], y=target, training_settings=training_settings
    )

    # Evaluate for the same diffusion coefficient used to generate training data.
    test_kappa = args.kappa
    u_fom, t = fom.solve(args.dt, args.end_time, test_kappa)
    u0_r = phi.T @ fom.u0

    rom = NnOpInfRom(model)
    u_rom_r, _ = rom.solve(u0_r, args.dt, args.end_time)
    u_rom = phi @ u_rom_r

    relative_error = np.linalg.norm(u_rom - u_fom) / np.linalg.norm(u_fom)
    print(f"Relative trajectory error (kappa={test_kappa:.3f}): {relative_error:.4e}")

    u_fom_final = u_fom[:, -1].reshape(args.grid_size, args.grid_size)
    u_rom_final = u_rom[:, -1].reshape(args.grid_size, args.grid_size)
    diff_final = np.abs(u_fom_final - u_rom_final)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    extent = (0.0, 1.0, 0.0, 1.0)
    vmin = min(np.min(u_fom_final), np.min(u_rom_final))
    vmax = max(np.max(u_fom_final), np.max(u_rom_final))

    im0 = axes[0].imshow(u_fom_final, origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("FOM final state")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    im1 = axes[1].imshow(u_rom_final, origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("NN-OpInf final state")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    im2 = axes[2].imshow(diff_final, origin="lower", extent=extent, cmap="magma")
    axes[2].set_title("Absolute error")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle(f"Transient heat diffusion, kappa={test_kappa:.2f}, rel. error={relative_error:.2e}")

    fig.savefig(os.path.join(output_dir, "heat_diffusion_solution.pdf"))
    fig.savefig(os.path.join(output_dir, "heat_diffusion_solution.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
