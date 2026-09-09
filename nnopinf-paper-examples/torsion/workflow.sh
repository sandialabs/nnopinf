#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"
julia_threads="${JULIA_THREADS:-1}"

families=(linear quadratic vanilla spsd-potential lpopinf linear-lagrangian)
dimensions=(4 8 16 32)

usage() {
    cat <<'EOF'
Usage:
  ./workflow.sh fom
  ./workflow.sh train [all|linear|quadratic|vanilla|spsd-potential|lpopinf|linear-lagrangian]
  ./workflow.sh simulate [family|all] [4|8|16|32|all]
  ./workflow.sh plot
  ./workflow.sh render [render_model_pngs.py options]
  ./workflow.sh all

Environment:
  NORMA_JL_ROOT  Required by fom, simulate, and all. Path to Norma.jl.
  PYTHON_BIN     Python executable to use (default: python).
  JULIA_THREADS  Julia thread count (default: 1).

The all stage runs fom, train all, simulate all all, and plot. Rendering is
excluded because it requires a separate ParaView/pvpython installation.
EOF
}

contains() {
    local needle="$1"
    shift
    local value
    for value in "$@"; do
        [[ "$value" == "$needle" ]] && return 0
    done
    return 1
}

require_norma() {
    if [[ -z "${NORMA_JL_ROOT:-}" ]]; then
        printf 'error: NORMA_JL_ROOT must point to a Norma.jl checkout\n' >&2
        exit 2
    fi
    if [[ ! -f "$NORMA_JL_ROOT/src/Norma.jl" ]]; then
        printf 'error: Norma entry point not found: %s/src/Norma.jl\n' "$NORMA_JL_ROOT" >&2
        exit 2
    fi
    command -v julia >/dev/null 2>&1 || {
        printf 'error: julia was not found on PATH\n' >&2
        exit 2
    }
}

run_norma() {
    local run_dir="$1"
    require_norma
    printf 'Running Norma in %s\n' "$run_dir"
    (
        cd -- "$run_dir"
        julia --threads "$julia_threads" \
            --project="$NORMA_JL_ROOT" \
            "$NORMA_JL_ROOT/src/Norma.jl" \
            torsion-in.yaml
    )
}

run_fom() {
    run_norma "$script_dir/fom"
}

training_script() {
    case "$1" in
        linear) echo "make_linear_opinf_models.py" ;;
        quadratic) echo "make_quadratic_opinf_models.py" ;;
        vanilla) echo "make_nn_models.py" ;;
        spsd-potential) echo "make_spsd_lagrangian_models.py" ;;
        lpopinf) echo "make_lpopinf_models.py" ;;
        linear-lagrangian) echo "make_linear_lagrangian_models.py" ;;
        *) return 1 ;;
    esac
}

train_one() {
    local family="$1"
    local training_file
    training_file="$(training_script "$family")"
    printf 'Training %s models\n' "$family"
    (
        cd -- "$script_dir/training"
        "$python_bin" "$training_file"
    )
}

run_train() {
    local selection="${1:-all}"
    local family
    if [[ "$selection" == all ]]; then
        for family in "${families[@]}"; do
            train_one "$family"
        done
    elif contains "$selection" "${families[@]}"; then
        train_one "$selection"
    else
        printf 'error: unknown model family: %s\n' "$selection" >&2
        exit 2
    fi
}

simulate_one() {
    local family="$1"
    local dimension="$2"
    local run_dir="$script_dir/roms/$family/dim$dimension"
    if [[ ! -f "$run_dir/torsion-in.yaml" ]]; then
        printf 'error: missing ROM configuration: %s/torsion-in.yaml\n' "$run_dir" >&2
        exit 2
    fi
    run_norma "$run_dir"
}

run_simulate() {
    local family_selection="${1:-all}"
    local dimension_selection="${2:-all}"
    local selected_families=()
    local selected_dimensions=()
    local family dimension

    if [[ "$family_selection" == all ]]; then
        selected_families=("${families[@]}")
    elif contains "$family_selection" "${families[@]}"; then
        selected_families=("$family_selection")
    else
        printf 'error: unknown model family: %s\n' "$family_selection" >&2
        exit 2
    fi

    if [[ "$dimension_selection" == all ]]; then
        selected_dimensions=("${dimensions[@]}")
    elif contains "$dimension_selection" "${dimensions[@]}"; then
        selected_dimensions=("$dimension_selection")
    else
        printf 'error: unsupported ROM dimension: %s\n' "$dimension_selection" >&2
        exit 2
    fi

    for family in "${selected_families[@]}"; do
        for dimension in "${selected_dimensions[@]}"; do
            simulate_one "$family" "$dimension"
        done
    done
}

run_plot() {
    printf 'Generating convergence plots\n'
    (
        cd -- "$script_dir/postprocess"
        "$python_bin" post_process.py
    )
}

run_render() {
    command -v pvpython >/dev/null 2>&1 || {
        printf 'error: pvpython was not found on PATH\n' >&2
        exit 2
    }
    shift
    (
        cd -- "$script_dir/postprocess"
        pvpython render_model_pngs.py "$@"
    )
}

stage="${1:-help}"
case "$stage" in
    fom)
        [[ "$#" -eq 1 ]] || { usage >&2; exit 2; }
        run_fom
        ;;
    train)
        [[ "$#" -le 2 ]] || { usage >&2; exit 2; }
        run_train "${2:-all}"
        ;;
    simulate)
        [[ "$#" -le 3 ]] || { usage >&2; exit 2; }
        run_simulate "${2:-all}" "${3:-all}"
        ;;
    plot)
        [[ "$#" -eq 1 ]] || { usage >&2; exit 2; }
        run_plot
        ;;
    render)
        run_render "$@"
        ;;
    all)
        [[ "$#" -eq 1 ]] || { usage >&2; exit 2; }
        run_fom
        run_train all
        run_simulate all all
        run_plot
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        printf 'error: unknown stage: %s\n' "$stage" >&2
        usage >&2
        exit 2
        ;;
esac
