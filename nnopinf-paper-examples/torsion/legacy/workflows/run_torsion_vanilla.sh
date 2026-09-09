#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
failed=0
sorted_dirs=$(for d in "${script_dir}"/dim*-vanilla/; do [[ -d "$d" ]] || continue; n="${d##*/dim}"; n="${n%-vanilla/}"; printf '%s\t%s\n' "$n" "$d"; done | sort -n | cut -f2-)
while IFS= read -r d; do
    [[ -n "$d" ]] || continue
    echo "Running vanilla NN model in $d"
    if (cd "$d" && julia --threads 1 --project=@/Users/ejparis/codes/Norma.jl /Users/ejparis/codes/Norma.jl/src/Norma.jl torsion-in.yaml); then :; else
        printf 'Run failed in %s\n' "$d" >&2
        failed=1
    fi
done <<< "$sorted_dirs"
exit "$failed"
