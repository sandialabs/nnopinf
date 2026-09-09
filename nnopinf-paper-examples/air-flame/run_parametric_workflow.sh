#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
configuration="${1:-parametric_opinf.yaml}"
python_bin="${PYTHON_BIN:-python}"
cd -- "$script_dir"
"$python_bin" make_uniform_parameters.py --i "$configuration"
"$python_bin" cdr_fom.py --i "$configuration"
mpirun -np "${MPI_PROCESSES:-1}" "$python_bin" ../src/build_ml_models.py --i "$configuration"
"$python_bin" simulate_roms.py --i "$configuration"
"$python_bin" make_plots.py --i "$configuration"
