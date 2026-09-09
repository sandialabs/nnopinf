"""Plot displacement convergence for the torsion ROM simulations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


AXIS_FONT = {"size": "20"}
plt.rcParams.update({"text.usetex": True, "font.family": "Serif"})

ROM_DIMS = [4, 8, 16, 32]
ROM_DIRECTORY = Path(__file__).resolve().parent.parent / "roms"
FOM_DIRECTORY = ROM_DIRECTORY.parent / "fom"

MODEL_RUNS = [
    ("P-OpInf-A", "linear", "green", "o"),
    ("P-OpInf-AH", "quadratic", "purple", "s"),
    ("NN-OpInf-SPSD-Potential", "spsd-potential", "red", "v"),
    ("NN-OpInf-NN", "vanilla", "blue", "^"),
    ("NN-OpInf-SpML-Lagrangian", "lpopinf", "orange", "*"),
    ("NN-OpInf-Linear-Lagrangian", "linear-lagrangian", "black", "P"),
]


def load_solution(solution_directory, max_snapshots=50, snapshot_stride=10):
    """Load displacement snapshots and their timestamps from one run."""
    solution_directory = Path(solution_directory)
    displacement_files = sorted(
        solution_directory.glob("torsion-in-disp-*.csv"),
        key=lambda path: int(path.stem.rsplit("-", 1)[1]),
    )[::snapshot_stride][:max_snapshots]
    if not displacement_files:
        raise FileNotFoundError(
            f"No displacement CSV files found in {solution_directory}"
        )

    solutions = [np.loadtxt(path, delimiter=",").T for path in displacement_files]
    timestamps = [
        np.loadtxt(
            solution_directory / path.name.replace("-disp-", "-time-"),
        )
        for path in displacement_files
    ]
    return np.stack(solutions, axis=2), np.asarray(timestamps)


def compute_relative_error(fom_solution, fom_times, rom_solution, rom_times, slice_end):
    """Compute an error over snapshots with matching timestamps."""
    fom_by_time = {float(time): index for index, time in enumerate(fom_times)}
    rom_fom_indices = []
    rom_indices = []
    for rom_index, time in enumerate(rom_times):
        fom_index = fom_by_time.get(float(time))
        if fom_index is not None:
            rom_fom_indices.append(fom_index)
            rom_indices.append(rom_index)

    n_snapshots = min(slice_end, len(rom_indices))
    if n_snapshots == 0:
        raise ValueError("No common displacement snapshots were found")

    fom_indices = rom_fom_indices[:n_snapshots]
    rom_indices = rom_indices[:n_snapshots]
    error = np.linalg.norm(
        rom_solution[..., rom_indices] - fom_solution[..., fom_indices]
    )
    denominator = np.linalg.norm(fom_solution[..., fom_indices]) + 1.0e-4
    return error / denominator, n_snapshots


def compute_errors_by_model(slice_end):
    """Compute convergence errors for every configured model and dimension."""
    fom, fom_times = load_solution(
        FOM_DIRECTORY,
        max_snapshots=slice_end,
        snapshot_stride=10,
    )
    errors = {}

    for label, directory_suffix, _color, _marker in MODEL_RUNS:
        model_errors = []
        for rom_dim in ROM_DIMS:
            rom_directory = ROM_DIRECTORY / directory_suffix / f"dim{rom_dim}"
            try:
                rom, rom_times = load_solution(
                    rom_directory,
                    max_snapshots=slice_end,
                    snapshot_stride=10,
                )
                error, n_snapshots = compute_relative_error(
                    fom, fom_times, rom, rom_times, slice_end
                )
            except (FileNotFoundError, OSError, ValueError) as exc:
                print(f"Skipping {label} (dimension {rom_dim}): {exc}")
                error = np.nan
                n_snapshots = 0

            if n_snapshots < slice_end:
                print(
                    f"Using {n_snapshots} of {slice_end} requested snapshots "
                    f"for {label} (dimension {rom_dim})"
                )
            model_errors.append(error)

        errors[label] = model_errors
        print(f"{label}: {model_errors}")

    return errors


def plot_errors(errors, filename):
    """Plot and save one convergence figure."""
    plt.figure()
    for label, _directory_suffix, color, marker in MODEL_RUNS:
        plt.plot(
            ROM_DIMS,
            errors[label],
            marker=marker,
            label=label,
            color=color,
            linewidth=2,
            markersize=11,
        )
    plt.xlabel(r"$\mathrm{Basis\ dimension}$", **AXIS_FONT)
    plt.ylabel(r"$\mathrm{Relative\ error}$", **AXIS_FONT)
    plt.ylim([5.0e-3, 2.0])
    plt.grid()
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    plot_errors(compute_errors_by_model(slice_end=25), "convergence.pdf")
    plot_errors(
        compute_errors_by_model(slice_end=50),
        "convergence-future-state.pdf",
    )


if __name__ == "__main__":
    main()
