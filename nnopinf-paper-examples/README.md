# NN-OpInf paper examples

This directory contains the Python and YAML sources used for the numerical
examples in the NN-OpInf paper. It is a source-focused copy of
`paper-examples/`: generated snapshots, trained models, and figures are not
included. The original directory is unchanged.

We provide this source code for reproducibility of the examples in the NN-OpInf paper.
We do not guarantee long term maintenance of these cases.

## Contents

- `src/`: shared model construction, integration, driver, and test code.
- `advection-diffusion/`: reproductive, energy-preserving, and parametric
  convection-diffusion-reaction experiments.
- `air-flame/`: parametric reacting-flow experiment.
- `burgers-example/`: reproductive, future-state, and ensemble-variance
  Burgers experiments.
- `heat-example/`: reproductive, future-state, parametric, and
  ensemble-variance nonlinear heat experiments.
- `torsion/`: staged solid-mechanics workflow, including its mesh and compact
  mass matrices.

Only compact inputs needed to launch the torsion case are copied. Runtime
outputs are written beside the corresponding configuration and are ignored by
the local `.gitignore`.

## Python installation

Python 3.13 is the tested interpreter. From the NN-OpInf repository root,
create an isolated environment and install the local package before the paper
dependencies:

```bash
python3.13 -m venv .venv-paper
source .venv-paper/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r nnopinf-paper-examples/requirements.txt
```

The requirements file pins the experiment-relevant packages, including the
base Git revisions of `normaopinf` and `romtools`. The torsion instructions
below apply the recorded local `normaopinf` changes and reinstall that checkout
in editable mode. ParaView is not installed by pip; its `pvpython` executable
is only needed to render the torsion Exodus files.
LaTeX is needed by the torsion convergence plotting script because its
Matplotlib configuration uses `text.usetex`.

The shell launchers use `python` by default. Set `PYTHON_BIN` to select another
interpreter. Air-flame model construction uses `mpirun`; set `MPI_PROCESSES`
to change its default of one process.

## First-order experiments

The launchers may be called from any directory. Each performs parameter
generation, FOM simulation, model construction, ROM simulation, and plotting
in that order.

```bash
# Advection-diffusion
./nnopinf-paper-examples/advection-diffusion/run_workflow.sh reproductive_opinf.yaml
./nnopinf-paper-examples/advection-diffusion/run_workflow.sh reproductive_opinf_ep.yaml
./nnopinf-paper-examples/advection-diffusion/run_workflow.sh parametric_opinf.yaml

# Burgers
./nnopinf-paper-examples/burgers-example/run_workflow.sh reproductive.yaml
./nnopinf-paper-examples/burgers-example/run_workflow.sh future.yaml
./nnopinf-paper-examples/burgers-example/run_variance_workflow.sh reproductive-variance.yaml
./nnopinf-paper-examples/burgers-example/run_variance_workflow.sh future-variance.yaml

# Nonlinear heat
./nnopinf-paper-examples/heat-example/run_workflow.sh reproductive.yaml
./nnopinf-paper-examples/heat-example/run_workflow.sh future.yaml
./nnopinf-paper-examples/heat-example/run_workflow.sh parametric.yaml
./nnopinf-paper-examples/heat-example/run_variance_workflow.sh reproductive-variance.yaml
./nnopinf-paper-examples/heat-example/run_variance_workflow.sh future-variance.yaml

# Air-flame
./nnopinf-paper-examples/air-flame/run_parametric_workflow.sh parametric_opinf.yaml
```

The source collection contains no reproductive or future-state air-flame YAML
file. The corresponding retained wrappers therefore require an explicit YAML
path and will print their usage if none is supplied.

Training neural models can take hours and uses the random and optimization
settings recorded in each YAML file. Output directories are set by each
configuration's `output-directory` entry.

## Torsion workflow

The torsion case is separated into FOM generation, model training, ROM
simulation, and post-processing:

```text
torsion/
├── input/          mesh
├── fom/            FOM configuration, compact mass matrix, and FOM output
├── training/       model-building scripts and generated models
├── roms/           ROM configurations grouped by family and dimension
├── postprocess/    convergence plots and optional ParaView rendering
└── legacy/         original alternate scripts retained for provenance
```

### Reproduce the Norma solver state

Two patches are provided because the torsion workflow used local changes in
both the Julia solver and its Python model-training package:

- `patches/norma-jl-601f1bea-local.patch` contains all 20 tracked Norma.jl
  modifications, the new mass-projection test, and the Julia 1.12.7
  `Manifest.toml` that pins the solver's transitive Julia dependencies.
- `patches/norma-opinf-cfc80bc9-local.patch` contains the four tracked
  `normaopinf` modifications and four new mass-matrix/regularization source
  and test files.

Apply them to clean checkouts at the recorded commits. `REPRO_ROOT` must point
to this NN-OpInf repository; the two dependency roots may be anywhere:

```bash
export REPRO_ROOT=/path/to/nnopinf
export NORMA_JL_ROOT=/path/to/Norma.jl
export NORMA_OPINF_ROOT=/path/to/norma-opinf

git clone https://github.com/sandialabs/Norma.jl "$NORMA_JL_ROOT"
git -C "$NORMA_JL_ROOT" checkout 601f1bea36d73cee8e67c2ab3c08cd4f162583ed
git -C "$NORMA_JL_ROOT" apply --check \
  "$REPRO_ROOT/nnopinf-paper-examples/patches/norma-jl-601f1bea-local.patch"
git -C "$NORMA_JL_ROOT" apply \
  "$REPRO_ROOT/nnopinf-paper-examples/patches/norma-jl-601f1bea-local.patch"
julia --project="$NORMA_JL_ROOT" -e 'using Pkg; Pkg.instantiate()'

git clone https://github.com/sandialabs/norma-opinf.git "$NORMA_OPINF_ROOT"
git -C "$NORMA_OPINF_ROOT" checkout cfc80bc9ccfa76bf4eaba939d1b300a682d63af3
git -C "$NORMA_OPINF_ROOT" apply --check \
  "$REPRO_ROOT/nnopinf-paper-examples/patches/norma-opinf-cfc80bc9-local.patch"
git -C "$NORMA_OPINF_ROOT" apply \
  "$REPRO_ROOT/nnopinf-paper-examples/patches/norma-opinf-cfc80bc9-local.patch"
python -m pip install -e "$NORMA_OPINF_ROOT"
```

Patch SHA-256 checksums are:

```text
9134857c5c0ed6eb36a2b355fc26f76e2ba81880322bd285ccf6d6ccfd5479b5  norma-jl-601f1bea-local.patch
3520b61ea37c22dcedefcbeba1613192ab91559005fea1a478482d6c0070ee12  norma-opinf-cfc80bc9-local.patch
```

The patches deliberately exclude editor swap files, credentials/settings
files, caches, package metadata, and archived or nested repository copies;
none are solver runtime inputs. Both patches were verified by applying them to
fresh archives of their base commits and byte-comparing every included file
with the local checkout.

### Run torsion

After preparing Norma.jl and `normaopinf`, use the staged workflow:

```bash
export NORMA_JL_ROOT=/path/to/Norma.jl
./nnopinf-paper-examples/torsion/workflow.sh fom
./nnopinf-paper-examples/torsion/workflow.sh train all
./nnopinf-paper-examples/torsion/workflow.sh simulate all all
./nnopinf-paper-examples/torsion/workflow.sh plot
```

The workflow supports selective runs:

```bash
# Train the four linear OpInf dimensions.
./nnopinf-paper-examples/torsion/workflow.sh train linear

# Simulate only the dimension-16 SPSD-potential model.
./nnopinf-paper-examples/torsion/workflow.sh simulate spsd-potential 16

# Run FOM, all training, all ROM simulations, and convergence plotting.
./nnopinf-paper-examples/torsion/workflow.sh all

# Optional Exodus rendering with ParaView.
./nnopinf-paper-examples/torsion/workflow.sh render \
  --disp-mag --times 0.0025,0.0050 --output-dir renders
```

Supported model families are `linear`, `quadratic`, `vanilla`,
`spsd-potential`, `lpopinf`, and `linear-lagrangian`; supported ROM dimensions
are 4, 8, 16, and 32. `JULIA_THREADS` defaults to 1. The `all` stage does not
run ParaView rendering.

## Recorded environment

These sources were inspected and packaged in the following environment on
2026-09-08:

- CPython 3.13.15, built with Apple Clang 21.0.0.
- pip 26.2.
- macOS 26.6.2 (build 25G83), ARM64.
- NN-OpInf 0.1.0 from this repository.
- Julia 1.12.7.
- Patched Norma.jl HEAD `601f1bea36d73cee8e67c2ab3c08cd4f162583ed`.
- Patched `normaopinf` base commit
  `cfc80bc9ccfa76bf4eaba939d1b300a682d63af3`.
- `romtools` commit `1c93a49efebcdc73acf8009c3eeed89b237d81e9`.


## Checks

Run the source-level tests from this directory:

```bash
cd nnopinf-paper-examples
PYTHONPATH=src pytest -q src
```

The numerical workflows are intentionally not part of the quick test suite
because they generate large data sets and train multiple neural ensembles.
